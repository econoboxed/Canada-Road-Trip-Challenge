#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute road travel times from one or more start points to Canadian regions.

Uses a local OSRM server, a border-crossings CSV, and optional region polygons
(Canada.geojson) to classify each start into a home region. When a home region is
known, its travel time is treated as zero (you are already there).

Supports:
  * Single-start mode (one start provided via CLI lat/lon)
  * Multi-start mode (multiple starts from a CSV)

Outputs a CSV of per-start, per-region travel times.
"""

import sys
import csv
import math
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
import time

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Shapely for region classification (Canada.geojson)
try:
    from shapely.geometry import shape, Point, Polygon, MultiPolygon  # type: ignore
    from shapely.prepared import prep  # type: ignore
    from shapely.validation import make_valid  # type: ignore
except Exception:
    shape = None
    Point = None
    Polygon = None
    MultiPolygon = None
    prep = None
    make_valid = None


EARTH_R = 6371000.0  # metres


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in metres."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2 * EARTH_R * math.asin(math.sqrt(a))



def _fmt_rate(done: int, start_ts: float) -> str:
    dt = max(1e-6, time.time() - start_ts)
    return f"{done/dt:,.1f}/s"


def _print_progress_prefix(prefix: str, processed: int, total: int, start_ts: float):
    pct = 100.0 * processed / total if total > 0 else 0.0
    rate = _fmt_rate(processed, start_ts)
    msg = f"{prefix} {processed:,}/{total:,} ({pct:5.1f}%)  |  {rate}"
    sys.stderr.write("\r" + msg[:120].ljust(120))
    sys.stderr.flush()



def _fix_poly(g):
    """Make invalid polygons valid, cheaply."""
    try:
        if g.is_valid:
            return g
        if make_valid is not None:
            return make_valid(g)
        return g.buffer(0)
    except Exception:
        return g


def load_regions(geojson_path: Path, section_field: str):
    """Load region polygons and their names from a GeoJSON file."""
    if shape is None:
        raise SystemExit(
            "[regions] Shapely is required to classify starts into regions; please install shapely."
        )

    with geojson_path.open("r", encoding="utf-8") as fp:
        feats = json.load(fp)

    polys: List[Polygon] = []
    names: List[str] = []

    for f in feats.get("features", []):
        props = f.get("properties", {}) or {}
        nm = str(
            props.get(section_field)
            or props.get("name")
            or props.get("NAME")
            or "unknown"
        )
        geom = f.get("geometry")
        if not geom:
            continue
        try:
            g = shape(geom)
        except Exception:
            continue
        if g.is_empty:
            continue

        if isinstance(g, MultiPolygon):
            for part in g.geoms:
                polys.append(_fix_poly(part))
                names.append(nm)
        elif isinstance(g, Polygon):
            polys.append(_fix_poly(g))
            names.append(nm)
        else:
            # Ignore non-area geometries
            continue

    if not polys:
        raise SystemExit(f"[regions] No polygons loaded from {geojson_path}")

    preps = [prep(p) if prep is not None else None for p in polys]

    # For logging: distinct region names
    uniq = []
    seen = set()
    for nm in names:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    sys.stderr.write(
        f"Loaded regions: \n"
        + ", ".join(uniq)
        + "\n\n"
    )
    sys.stderr.flush()

    return polys, names, preps


def region_for_point(
    lat: float,
    lon: float,
    polys: List[Polygon],
    names: List[str],
    preps,
) -> Optional[str]:
    """Return the name of the first region whose polygon contains the point."""
    if Point is None:
        return None
    pt = Point(lon, lat)
    for i, poly in enumerate(polys):
        try:
            if preps[i] is not None:
                if preps[i].contains(pt):
                    return names[i]
            else:
                if poly.contains(pt):
                    return names[i]
        except Exception:
            continue
    return None



def load_border_points(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Load border crossing points into a mapping: region_name -> [(lat, lon), ...]."""
    mapping: Dict[str, List[Tuple[float, float]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise SystemExit(f"[borders] CSV {csv_path} has no header row.")

        for row in rd:
            fp = (
                row.get("from_section")
                or row.get("from_province")
                or row.get("from_region")
                or row.get("from")
                or ""
            ).strip()
            tp = (
                row.get("to_section")
                or row.get("to_province")
                or row.get("to_region")
                or row.get("to")
                or ""
            ).strip()

            lat_s = row.get("latitude") or row.get("lat")
            lon_s = row.get("longitude") or row.get("lon")

            if not lat_s or not lon_s:
                continue

            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except Exception:
                continue

            if fp:
                mapping.setdefault(fp, []).append((lat, lon))
            if tp:
                mapping.setdefault(tp, []).append((lat, lon))

    if not mapping:
        raise SystemExit(f"[borders] No usable border points found in {csv_path}")

    return mapping



def load_starts_csv(csv_path: Path) -> List[Tuple[str, float, float]]:
    """Load start points from a CSV with columns name, lat, lon (or variants)."""
    starts: List[Tuple[str, float, float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise SystemExit(f"[starts] CSV {csv_path} has no header row.")

        for row in rd:
            name = (row.get("name") or row.get("label") or "").strip() or "start"
            lat_s = row.get("lat") or row.get("latitude")
            lon_s = row.get("lon") or row.get("longitude")
            if not lat_s or not lon_s:
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except Exception:
                continue
            starts.append((name, lat, lon))

    if not starts:
        raise SystemExit(f"[starts] No usable start points found in {csv_path}")

    return starts



def osrm_route(
    osrm_url: str,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    timeout: float = 20.0,
) -> Tuple[float, float, float, float, float, float]:
    """Call OSRM and return (distance_km, duration_s, snapped_start_lat, snapped_start_lon, snapped_end_lat, snapped_end_lon)."""
    url = (
        osrm_url.rstrip("/")
        + f"/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=false"
    )

    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"OSRM request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"OSRM HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"OSRM JSON decode failed: {e}")

    code = data.get("code")
    if code != "Ok":
        raise RuntimeError(f"OSRM error code: {code}")

    routes = data.get("routes") or []
    if not routes:
        raise RuntimeError("OSRM returned no routes")

    route = routes[0]
    dist_m = float(route.get("distance", 0.0))
    dur_s = float(route.get("duration", 0.0))

    waypoints = data.get("waypoints") or []
    snapped_start_lon = start_lon
    snapped_start_lat = start_lat
    snapped_end_lon = end_lon
    snapped_end_lat = end_lat
    if len(waypoints) >= 1:
        loc0 = waypoints[0].get("location", None)
        if isinstance(loc0, (list, tuple)) and len(loc0) == 2:
            try:
                snapped_start_lon = float(loc0[0])
                snapped_start_lat = float(loc0[1])
            except Exception:
                pass
    if len(waypoints) >= 2:
        loc1 = waypoints[1].get("location", None)
        if isinstance(loc1, (list, tuple)) and len(loc1) == 2:
            try:
                snapped_end_lon = float(loc1[0])
                snapped_end_lat = float(loc1[1])
            except Exception:
                pass

    return dist_m / 1000.0, dur_s, snapped_start_lat, snapped_start_lon, snapped_end_lat, snapped_end_lon


def format_time_h_m(t_sec: float) -> str:
    """Format seconds as 'Hh MMm'."""
    if not isinstance(t_sec, (int, float)) or math.isnan(t_sec):
        return ""
    t_h = t_sec / 3600.0
    h = int(t_h)
    m = int(round((t_h - h) * 60.0))
    if m == 60:
        h += 1
        m = 0
    return f"{h}h {m:02d}m"



def compute_times_for_start(
    start_name: str,
    start_lat: float,
    start_lon: float,
    border_map: Dict[str, List[Tuple[float, float]]],
    osrm_url: str,
    home_region: Optional[str] = None,
    snap_max_km: float = 50.0,
    start_snap_max_km: float = 50.0,
) -> Dict[str, Tuple[float, float, float, float, str, float, float]]:
    """Compute best travel times from one start to each region; returns a mapping region_name -> metrics tuple."""
    results: Dict[str, Tuple[float, float, float, float, str, float, float]] = {}

    regions = sorted(border_map.keys())

    total_regions = len(regions)
    total_points = sum(len(border_map[nm]) for nm in regions)
    processed_points = 0
    done_regions = 0
    start_snap_delta_km_first: Optional[float] = None
    unreachable_regions: List[str] = []
    
    use_tqdm = (tqdm is not None) and sys.stderr.isatty() and total_points > 0
    checks_bar = None
    if use_tqdm:
        checks_bar = tqdm(
            total=total_points,
            unit="check",
            ncols=100,
            desc=start_name,
            leave=False,
        )

    for nm in regions:
        done_regions += 1
        cand_points = border_map[nm]

        # If this region is the start's home region, force 0 travel time and skip OSRM.
        if home_region and nm == home_region:
            if checks_bar is not None:
                checks_bar.update(len(cand_points))
            results[nm] = (
                0.0,
                0.0,
                0.0,
                0.0,
                "0h 00m",
                start_lat,
                start_lon,
            )
            continue
            
        best_t_sec: Optional[float] = None
        best_road_km: float = math.nan
        best_lat: float = math.nan
        best_lon: float = math.nan
        best_straight_km: float = math.nan

        for idx, (cand_lat, cand_lon) in enumerate(cand_points, start=1):
            processed_points += 1
            
            if checks_bar is not None:
                checks_bar.update(1)

            straight_km = haversine_m(start_lat, start_lon, cand_lat, cand_lon) / 1000.0
            try:
                road_km_c, t_sec_c, s_start_lat, s_start_lon, s_lat, s_lon = osrm_route(
                    osrm_url,
                    start_lon,
                    start_lat,
                    cand_lon,
                    cand_lat,
                )
            except Exception:
                continue

            # Check how far OSRM snapped the *start* point onto the road graph.
            start_snap_delta_km = (
                haversine_m(start_lat, start_lon, s_start_lat, s_start_lon) / 1000.0
            )
            if start_snap_delta_km_first is None:
                start_snap_delta_km_first = start_snap_delta_km
                if start_snap_delta_km_first > start_snap_max_km:
                    warn_msg = (
                        f"[warn] Start '{start_name}' appears "
                        f"{start_snap_delta_km_first:.1f} km from nearest road; "
                        "skipping road routing for this start."
                    )
                    colored = f"\033[33m{warn_msg}\033[0m"
                    if checks_bar is not None:
                        checks_bar.write(colored)
                        checks_bar.close()
                    else:
                        sys.stderr.write(colored + "\n")
                        sys.stderr.flush()
                    if home_region and home_region in results:
                        return {home_region: results[home_region]}
                    return {}

            # Sanity-check snap distance at the *end* (border point) to guard against silly ferry midpoints.
            snap_delta_km = haversine_m(cand_lat, cand_lon, s_lat, s_lon) / 1000.0
            if snap_delta_km > snap_max_km:
                # Ignore this candidate; only warn if all fail
                continue

            if best_t_sec is None or t_sec_c < best_t_sec:
                best_t_sec = t_sec_c
                best_road_km = road_km_c
                best_lat = cand_lat   # keep logical border-crossing coordinate from CSV
                best_lon = cand_lon
                best_straight_km = straight_km

        if best_t_sec is None:
            unreachable_regions.append(nm)
            results[nm] = (math.nan, math.nan, math.nan, math.nan, "", math.nan, math.nan)
        else:
            t_h = best_t_sec / 3600.0
            t_str = format_time_h_m(best_t_sec)
            results[nm] = (
                best_straight_km,
                best_road_km,
                best_t_sec,
                t_h,
                t_str,
                best_lat,
                best_lon,
            )
            
    if unreachable_regions:
        regions_str = ", ".join(unreachable_regions)
        warn_msg = f"[warn] No successful OSRM route found from '{start_name}' to: {regions_str}"
        colored = f"\033[33m{warn_msg}\033[0m"
        if checks_bar is not None:
            checks_bar.write(colored)
        else:
            sys.stderr.write(colored + "\n")
            sys.stderr.flush()
    if checks_bar is not None:
        checks_bar.close()
    return results

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Travel times from one or more start points to each region in a border-crossings CSV, "
            "using a local OSRM routing server."
        )
    )
    ap.add_argument(
        "--borders-csv",
        required=True,
        help="CSV of inter-regional crossings (from_section,to_section,latitude,longitude,...)",
    )
    ap.add_argument(
        "--regions",
        help="Canada.geojson-style polygons used to detect which region a start lies in",
    )
    ap.add_argument(
        "--section-field",
        default="CDNAME",
        help="Property name in regions GeoJSON for region names (e.g. CDNAME or CNAME)",
    )
    ap.add_argument(
        "--start-lat",
        type=float,
        help="Start latitude (e.g. 44.6488 for Halifax, NS)",
    )
    ap.add_argument(
        "--start-lon",
        type=float,
        help="Start longitude (e.g. -63.5753 for Halifax, NS)",
    )
    ap.add_argument(
        "--start-name",
        default="Start",
        help="Label for the start point (for output)",
    )
    ap.add_argument(
        "--starts-csv",
        help="CSV of multiple start points (columns: name,lat,lon)",
    )
    ap.add_argument(
        "--osrm-url",
        default="http://127.0.0.1:5001",
        help="Base URL of the OSRM server (default http://127.0.0.1:5001)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV (single-start: per-region times; multi-start: per-start per-region times)",
    )

    args = ap.parse_args()

    borders_path = Path(args.borders_csv)
    out_csv = Path(args.out)

    if not borders_path.exists():
        raise SystemExit(f"[borders] File not found: {borders_path}")

    # Optional regions for classifying starts
    polys = None
    region_names = None
    preps = None
    if args.regions:
        regions_path = Path(args.regions)
        if not regions_path.exists():
            sys.stderr.write(f"[regions] WARNING: regions file not found: {regions_path}, skipping start classification\n")
            sys.stderr.flush()
        else:
            polys, region_names, preps = load_regions(regions_path, args.section_field)

    def detect_home_region(name: str, lat: float, lon: float, bar=None) -> Optional[str]:
        if polys is None or region_names is None:
            return None
        nm = region_for_point(lat, lon, polys, region_names, preps)
        if nm is None:
            # Print warning on its own line, in yellow, without corrupting tqdm bars
            msg = f"\033[33m[warn] Start '{name}' not inside any region polygon\033[0m"
            if bar is not None and tqdm is not None and sys.stderr.isatty():
                # Use tqdm-safe logging so the Locations bar is temporarily moved, then redrawn
                bar.write(msg)
            else:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
        return nm

    border_map = load_border_points(borders_path)
    multi_mode = args.starts_csv is not None

    if multi_mode:
        starts_path = Path(args.starts_csv)
        if not starts_path.exists():
            raise SystemExit(f"[starts] File not found: {starts_path}")

        starts = load_starts_csv(starts_path)

        total_starts = len(starts)
        use_tqdm_starts = (tqdm is not None) and sys.stderr.isatty() and total_starts > 0
        starts_bar = None
        if use_tqdm_starts:
            starts_bar = tqdm(
                total=total_starts,
                unit="start",
                ncols=100,
                desc="Locations",
                leave=True,
            )

        all_rows = []
        avg_for_csv = []
        for idx, (name, s_lat, s_lon) in enumerate(starts, start=1):
            home_region = detect_home_region(name, s_lat, s_lon, starts_bar)

            if starts_bar is not None:
                starts_bar.set_postfix_str(name)

            res = compute_times_for_start(
                name,
                s_lat,
                s_lon,
                border_map,
                args.osrm_url,
                home_region=home_region,
            )

            if starts_bar is not None:
                starts_bar.update(1)

            rows_for_start = []
            for nm, (straight_km, road_km, t_sec, t_h, t_str, end_lat, end_lon) in res.items():
                if isinstance(t_sec, (int, float)) and not math.isnan(t_sec):
                    rows_for_start.append(
                        (
                            name,
                            nm,
                            straight_km,
                            road_km,
                            t_sec,
                            t_h,
                            t_str,
                            end_lat,
                            end_lon,
                        )
                    )

            def _sort_key_start(r):
                return r[4]  # t_sec

            rows_for_start.sort(key=_sort_key_start)

            if rows_for_start:
                n_regions = len(rows_for_start)
                avg_sec_all = sum(r[4] for r in rows_for_start) / n_regions
                avg_str_all = format_time_h_m(avg_sec_all)

                all_rows.extend(rows_for_start)

                per_k_hm = {}
                for k in range(20, 2, -1):  # 20, 19, ..., 3
                    if n_regions >= k:
                        subset = rows_for_start[:k]
                        avg_sec_k = sum(r[4] for r in subset) / k
                        per_k_hm[k] = format_time_h_m(avg_sec_k)
                    else:
                        per_k_hm[k] = ""

                avg_for_csv.append(
                    (name, n_regions, avg_sec_all, avg_str_all, per_k_hm)
                )
                
        if starts_bar is not None:
            starts_bar.close()
            
        avg_csv = out_csv.parent / "averages.csv"
        with avg_csv.open("w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)

            # Header with explicit columns for averages over the fastest 20..3 regions
            base_header = [
                "from_name",
                "n_regions",
                "avg_time_sec",
                "avg_time_h_m",
            ]
            k_range = list(range(20, 3 - 1, -1))  # 20, 19, ..., 3
            k_header = [f"avg_top_{k}_h_m" for k in k_range]
            wr.writerow(base_header + k_header)

            # Per-location rows only
            for name, n_regions, avg_sec_all, avg_str_all, per_k_hm in avg_for_csv:
                row = [
                    name,
                    n_regions,
                    f"{avg_sec_all:.1f}",
                    avg_str_all,
                ]
                row.extend(per_k_hm.get(k, "") for k in k_range)
                wr.writerow(row)

        with out_csv.open("w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)
            wr.writerow([
                "from_name",
                "to_region",
                "straight_distance_km",
                "road_distance_km",
                "road_time_sec",
                "road_time_hours",
                "road_time_h_m",
                "end_lat",
                "end_lon",
            ])
            for (from_name, nm, straight_km, road_km, t_sec, t_h, t_str, end_lat, end_lon) in all_rows:
                wr.writerow([
                    from_name,
                    nm,
                    f"{straight_km:.1f}" if isinstance(straight_km, (int, float)) and not math.isnan(straight_km) else "",
                    f"{road_km:.1f}" if isinstance(road_km, (int, float)) and not math.isnan(road_km) else "",
                    f"{t_sec:.1f}" if isinstance(t_sec, (int, float)) and not math.isnan(t_sec) else "",
                    f"{t_h:.3f}" if isinstance(t_h, (int, float)) and not math.isnan(t_h) else "",
                    t_str,
                    f"{end_lat:.7f}" if isinstance(end_lat, (int, float)) and not math.isnan(end_lat) else "",
                    f"{end_lon:.7f}" if isinstance(end_lon, (int, float)) and not math.isnan(end_lon) else "",
                ])

        print(f"Done. Wrote per-start road-based travel-time table to {out_csv}")
        return

    if args.start_lat is None or args.start_lon is None:
        raise SystemExit(
            "In single-start mode you must provide --start-lat and --start-lon (or use --starts-csv)."
        )

    start_lat = args.start_lat
    start_lon = args.start_lon

    home_region = detect_home_region(args.start_name, start_lat, start_lon)

    res = compute_times_for_start(
        args.start_name,
        start_lat,
        start_lon,
        border_map,
        args.osrm_url,
        home_region=home_region,
    )

    rows = []
    for nm, (
        straight_km,
        road_km,
        t_sec,
        t_h,
        t_str,
        end_lat,
        end_lon,
    ) in res.items():
        rows.append(
            (
                args.start_name,
                nm,
                straight_km,
                road_km,
                t_sec,
                t_h,
                t_str,
                end_lat,
                end_lon,
            )
        )

    def _sort_key_single(r):
        t_sec = r[4]
        if not isinstance(t_sec, (int, float)) or math.isnan(t_sec):
            return math.inf
        return t_sec

    rows.sort(key=_sort_key_single)

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(
            [
                "from_name",
                "to_region",
                "straight_distance_km",
                "road_distance_km",
                "road_time_sec",
                "road_time_hours",
                "road_time_h_m",
                "end_lat",
                "end_lon",
            ]
        )
        for (
            from_name,
            nm,
            straight_km,
            road_km,
            t_sec,
            t_h,
            t_str,
            end_lat,
            end_lon,
        ) in rows:
            wr.writerow(
                [
                    from_name,
                    nm,
                    f"{straight_km:.1f}" if isinstance(straight_km, (int, float)) and not math.isnan(straight_km) else "",
                    f"{road_km:.1f}" if isinstance(road_km, (int, float)) and not math.isnan(road_km) else "",
                    f"{t_sec:.1f}" if isinstance(t_sec, (int, float)) and not math.isnan(t_sec) else "",
                    f"{t_h:.3f}" if isinstance(t_h, (int, float)) and not math.isnan(t_h) else "",
                    t_str,
                    f"{end_lat:.7f}" if isinstance(end_lat, (int, float)) and not math.isnan(end_lat) else "",
                    f"{end_lon:.7f}" if isinstance(end_lon, (int, float)) and not math.isnan(end_lon) else "",
                ]
            )

    print(f"Done. Wrote OSRM-based travel-time table to {out_csv}")


if __name__ == "__main__":
    main()

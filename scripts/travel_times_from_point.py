#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute travel times from one or more start points (e.g. Halifax, NS, Vancouver, etc.)
to each region appearing in a border-crossings CSV, using a local OSRM server.

Two modes:

  1) Single-start mode (backwards-compatible):
       python3 travel_times_from_point.py \
         --borders-csv boundary_crossings.csv \
         --start-lat 44.6488 \
         --start-lon -63.5753 \
         --start-name "Halifax, NS" \
         --osrm-url http://127.0.0.1:5001 \
         --out times_from_halifax_border.csv

     Output: one row per region, with distances/times from that single start.

  2) Multi-start mode (NEW):
       python3 travel_times_from_point.py \
         --borders-csv boundary_crossings.csv \
         --starts-csv starts.csv \
         --osrm-url http://127.0.0.1:5001 \
         --out avg_times_by_region.csv

     where starts.csv has columns: name,lat,lon

     Output: one row per (start, region) pair, like single-start mode but repeated for each start, plus a final summary row per start giving the average road time from that start to all reachable regions (to_region="__AVERAGE__").

Border CSV format:
  - expected columns (any of these variants):
        from_section / from_province / from_region / from
        to_section   / to_province   / to_region   / to
        latitude     / lat
        longitude    / lon

Each border crossing point is associated with both regions that share it.
"""

import sys
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
import time
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# --------------------------------------------------------------------
# Distance helper
# --------------------------------------------------------------------

EARTH_R = 6371000.0  # metres


# --------------------------------------------------------------------
# Progress helpers
# --------------------------------------------------------------------

def _fmt_rate(done: int, start_ts: float) -> str:
    dt = max(1e-6, time.time() - start_ts)
    return f"{done/dt:,.1f}/s"

def _print_progress_prefix(prefix: str, processed: int, total: int, start_ts: float):
    pct = 100.0 * processed / total if total > 0 else 0.0
    rate = _fmt_rate(processed, start_ts)
    msg = f"{prefix} {processed:,}/{total:,} ({pct:5.1f}%)  |  {rate}"
    sys.stderr.write("\r" + msg[:120].ljust(120))
    sys.stderr.flush()


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


# --------------------------------------------------------------------
# Border crossings CSV loader
# --------------------------------------------------------------------


def load_border_points(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Load border crossing points from a CSV into a map:
         region_name -> [(lat, lon), ...]

    The CSV is expected to have columns like (any of these variants):

        from_section / from_province / from_region / from
        to_section   / to_province   / to_region   / to
        latitude     / lat
        longitude    / lon

    Each crossing point is associated with both regions that share it.
    """
    mapping: Dict[str, List[Tuple[float, float]]] = {}

    # First pass: count data rows for progress display
    total_rows = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for _ in f:
            total_rows += 1
    if total_rows > 0:
        total_rows -= 1  # subtract header

    start_ts = time.time()
    use_tqdm = (tqdm is not None) and sys.stderr.isatty() and total_rows > 0
    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=total_rows,
            unit="row",
            ncols=100,
            desc=f"Reading {csv_path.name}",
            leave=True,
        )

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise SystemExit(f"[borders] CSV {csv_path} has no header row.")

        row_idx = 0
        for row in rd:
            row_idx += 1
            if pbar is not None:
                pbar.update(1)
            else:
                if total_rows > 0 and row_idx % 1000 == 0:
                    _print_progress_prefix("[borders] loading border points...", row_idx, total_rows, start_ts)

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

        if pbar is not None:
            pbar.close()
        elif total_rows > 0:
            _print_progress_prefix("[borders] loading border points...", row_idx, total_rows, start_ts)
            sys.stderr.write("\n")
            sys.stderr.flush()

    if not mapping:
        raise SystemExit(f"[borders] No usable border points found in {csv_path}")

    sys.stderr.write(
        f"[borders] Loaded border points for {len(mapping)} regions from {csv_path}\n"
    )
    sys.stderr.flush()
    return mapping


# --------------------------------------------------------------------
# Starts CSV loader (multi-start mode)
# --------------------------------------------------------------------


def load_starts_csv(csv_path: Path) -> List[Tuple[str, float, float]]:
    """Load multiple start points from a CSV.

    Expected columns:
      - name (or label)
      - lat or latitude
      - lon or longitude
    """
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

    sys.stderr.write(f"[starts] Loaded {len(starts)} start points from {csv_path}\n")
    sys.stderr.flush()
    return starts


# --------------------------------------------------------------------
# OSRM routing helper
# --------------------------------------------------------------------


def osrm_route(
    osrm_url: str,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    timeout: float = 20.0,
) -> Tuple[float, float, float, float, float, float]:
    """Query OSRM for a driving route between two points.

    Returns:
      (distance_km, duration_s, snapped_start_lat, snapped_start_lon, snapped_end_lat, snapped_end_lon).

    Raises RuntimeError on failure.
    """
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


# --------------------------------------------------------------------
# Per-start computation helper
# --------------------------------------------------------------------

def compute_times_for_start(
    start_name: str,
    start_lat: float,
    start_lon: float,
    border_map: Dict[str, List[Tuple[float, float]]],
    osrm_url: str,
    snap_max_km: float = 50.0,
    start_snap_max_km: float = 50.0,
) -> Dict[str, Tuple[float, float, float, float, str, float, float]]:
    """Compute best travel times from a single start to each region.

    Returns a dict:
      region_name -> (
          straight_km,
          road_km,
          t_sec,
          t_h,
          t_str,
          end_lat,
          end_lon,
      )

    If no route is found for a region, t_sec will be NaN and strings empty.
    """
    results: Dict[str, Tuple[float, float, float, float, str, float, float]] = {}

    regions = sorted(border_map.keys())

    total_regions = len(regions)
    # Total number of candidate border points across all regions
    total_points = sum(len(border_map[nm]) for nm in regions)
    processed_points = 0

    # Fancy progress bar (tqdm) if available + TTY; else manual single-line progress
    use_tqdm = (tqdm is not None) and sys.stderr.isatty() and total_points > 0
    pbar = None
    start_ts = time.time()

    if use_tqdm:
        pbar = tqdm(
            total=total_points,
            unit="pt",
            ncols=100,
            desc=f"osrm {start_name}",
            leave=True,
        )

    done_regions = 0
    start_snap_delta_km_first: Optional[float] = None

    for nm in regions:
        done_regions += 1

        cand_points = border_map[nm]

        best_t_sec: Optional[float] = None
        best_road_km: float = math.nan
        best_lat: float = math.nan
        best_lon: float = math.nan
        best_straight_km: float = math.nan

        for idx, (cand_lat, cand_lon) in enumerate(cand_points, start=1):
            # Count this candidate as “processed” for progress purposes
            processed_points += 1

            if use_tqdm:
                # tqdm bar: percent comes from processed_points / total_points
                pbar.update(1)
                # show how many regions we’ve worked through in the postfix
                if processed_points % 10 == 0 or processed_points == total_points:
                    pbar.set_postfix(regions=f"{done_regions}/{total_regions}")
            else:
                # Manual single-line progress when tqdm isn't active
                if total_points > 0 and processed_points % 20 == 0:
                    pct = 100.0 * processed_points / total_points
                    sys.stderr.write(
                        f"\r[osrm] {start_name}: {done_regions}/{total_regions} regions ({pct:5.1f}%)"
                    )
                    sys.stderr.flush()

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
                # Only warn later if *all* candidates fail
                continue

            # Check how far OSRM snapped the *start* point onto the road graph.
            start_snap_delta_km = haversine_m(start_lat, start_lon, s_start_lat, s_start_lon) / 1000.0
            if start_snap_delta_km_first is None:
                start_snap_delta_km_first = start_snap_delta_km
                if start_snap_delta_km_first > start_snap_max_km:
                    # This start is effectively off the drivable network; abort this start entirely.
                    sys.stderr.write(
                        f"\n[warn] Start '{start_name}' appears {start_snap_delta_km_first:.1f} km from nearest road; skipping road routing for this start.\n"
                    )
                    sys.stderr.flush()
                    return {}

            # Sanity-check snap distance at the *end* (border point) to guard against silly ferry midpoints.
            snap_delta_km = haversine_m(cand_lat, cand_lon, s_lat, s_lon) / 1000.0
            if snap_delta_km > snap_max_km:
                # Ignore this candidate; only warn if all fail
                continue

            if best_t_sec is None or t_sec_c < best_t_sec:
                best_t_sec = t_sec_c
                best_road_km = road_km_c
                # Keep logical border-crossing coordinate from CSV
                best_lat = cand_lat
                best_lon = cand_lon
                best_straight_km = straight_km

        if best_t_sec is None:
            sys.stderr.write(
                f"\n[warn] No successful OSRM route found for region '{nm}' from '{start_name}'\n"
            )
            sys.stderr.flush()
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

    # Finish progress display nicely
    if use_tqdm and pbar is not None:
        # tqdm already showed a nice progress bar; just close it.
        pbar.close()
    elif total_points > 0:
        # Fallback manual summary when tqdm is not used
        pct = 100.0 * processed_points / total_points
        sys.stderr.write(
            f"\r[osrm] {start_name}: {total_regions}/{total_regions} regions ({pct:5.1f}%)\n"
        )
        sys.stderr.flush()

    return results
# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


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
    # Single-start options (backwards compatible)
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
    # Multi-start mode
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
        help="Output CSV (single-start: per-region times; multi-start: average per region)",
    )

    args = ap.parse_args()

    borders_path = Path(args.borders_csv)
    out_csv = Path(args.out)

    if not borders_path.exists():
        raise SystemExit(f"[borders] File not found: {borders_path}")

    # 1) Load border points
    border_map = load_border_points(borders_path)

    # Decide mode: multi-start vs single-start
    multi_mode = args.starts_csv is not None

    if multi_mode:
        # ----------------------
        # Multi-start mode
        # ----------------------
        starts_path = Path(args.starts_csv)
        if not starts_path.exists():
            raise SystemExit(f"[starts] File not found: {starts_path}")

        starts = load_starts_csv(starts_path)

        total_starts = len(starts)
        sys.stderr.write(f"[info] Running OSRM routing for {total_starts} start point(s)\n")
        sys.stderr.flush()

        all_rows = []
        avg_times_summary = []
        for idx, (name, s_lat, s_lon) in enumerate(starts, start=1):
            sys.stderr.write(
                f"[info] Start {idx}/{total_starts}: {name} at ({s_lat:.4f}, {s_lon:.4f}), OSRM {args.osrm_url}\n"
            )
            sys.stderr.flush()

            res = compute_times_for_start(
                name,
                s_lat,
                s_lon,
                border_map,
                args.osrm_url,
            )

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
            # sort rows for this start by increasing road time (NaNs already filtered)
            def _sort_key_start(r):
                return r[4]  # t_sec
            rows_for_start.sort(key=_sort_key_start)

            if rows_for_start:
                # Total reachable regions for this start
                n_regions = len(rows_for_start)

                # Average over all reachable regions (used for global best/worst summary)
                avg_sec_all = sum(r[4] for r in rows_for_start) / n_regions
                avg_str_all = format_time_h_m(avg_sec_all)

                # Append detailed rows for this start only (no summary rows in CSV)
                all_rows.extend(rows_for_start)

                # Track per-start averages for global summary (still based on ALL reachable regions)
                avg_times_summary.append((name, avg_sec_all, avg_str_all))

                # Compute prefix averages for fastest k regions, for k = N, N-1, ..., 3
                # rows_for_start is already sorted by increasing travel time.
                prefix_parts = []
                # We already have the average for all N; start from N-1 down to 3
                for k in range(n_regions - 1, 2, -1):
                    subset = rows_for_start[:k]
                    avg_sec_k = sum(r[4] for r in subset) / k
                    avg_str_k = format_time_h_m(avg_sec_k)
                    prefix_parts.append(f"{k}:{avg_str_k}")

                # Build a compact one-line summary for this start
                if prefix_parts:
                    prefix_str = ", ".join(prefix_parts)
                    msg = (
                        f"[avg] {name}: all {n_regions}: {avg_str_all} | "
                        f"fastest-k averages (k:time) -> {prefix_str}\n"
                    )
                else:
                    # Fallback when very few regions are reachable (<= 3)
                    msg = f"[avg] {name}: all {n_regions}: {avg_str_all}\n"

                sys.stderr.write(msg)
                sys.stderr.flush()

        if avg_times_summary:
            best = min(avg_times_summary, key=lambda x: x[1])
            worst = max(avg_times_summary, key=lambda x: x[1])
            sys.stderr.write(
                f"[summary] Lowest avg: {best[0]} ({best[2]}), Highest avg: {worst[0]} ({worst[2]})\n"
            )
            sys.stderr.flush()

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

    # ----------------------
    # Single-start mode (original behaviour)
    # ----------------------
    if args.start_lat is None or args.start_lon is None:
        raise SystemExit(
            "In single-start mode you must provide --start-lat and --start-lon (or use --starts-csv)."
        )

    start_lat = args.start_lat
    start_lon = args.start_lon

    sys.stderr.write(
        f"[info] Start: {args.start_name} at ({start_lat:.4f}, {start_lon:.4f}), OSRM {args.osrm_url}\n"
    )
    sys.stderr.flush()

    res = compute_times_for_start(
        args.start_name,
        start_lat,
        start_lon,
        border_map,
        args.osrm_url,
    )

    # Turn dict into sortable rows
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

    # Sort by increasing road time (NaNs last)
    def _sort_key_single(r):
        t_sec = r[4]
        if not isinstance(t_sec, (int, float)) or math.isnan(t_sec):
            return math.inf
        return t_sec

    rows.sort(key=_sort_key_single)

    # Write per-region CSV
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

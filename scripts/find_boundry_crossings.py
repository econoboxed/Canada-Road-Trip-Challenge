#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, json, csv, math, argparse, time
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

from shapely.geometry import shape, Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
try:
    from shapely.prepared import prep
except Exception:
    prep = None  # Shapely < 1.6 fallback
try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None
from shapely.ops import nearest_points as _nearest_points

# Optional pretty progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# ---------------------------
# Config defaults (can be overridden by CLI)
# ---------------------------
DEFAULT_REGIONS = "Canada.geojson"
DEFAULT_SECTION_FIELD = "CNAME"
DEFAULT_ROADS = "roads_ferries.geojsonl"
DEFAULT_OUT = "boundary_crossings.csv"

# rejects output chunking (split every N rows)
REJECTS_CHUNK_DEFAULT = 2500

# drivable filters (base allowlist)
DRIVABLE = {
    "motorway","trunk","primary","secondary","tertiary",
    "unclassified","residential","service",
    "motorway_link","trunk_link","primary_link","secondary_link","tertiary_link",
    "living_street","track", "road"  # explicitly car-legal
}

# candidates that might be car-drivable depending on tags

# explicit drops (always pedestrian/non-motor)
NON_MOTOR = {"footway","path","cycleway","bridleway","steps","pedestrian"}

# sampling + clustering
BASE_SAMPLES = 31            # max samples per line (adaptive below)
SAMPLE_PER_METRES = 600.0    # ~1 point per N metres
CLUSTER_RADIUS_M = 60.0
EARTH_R = 6371000.0

# near-shore classification thresholds (km)
SHORE_KM_FERRY  = 100.0   # long sea routes (e.g., NS↔NL)
SHORE_KM_BRIDGE = 20.0    # bridges/causeways over water

# ---------------------------
# Helpers
# ---------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*EARTH_R*math.asin(math.sqrt(a))

# ---------------------------
# Regions loading
# ---------------------------

def _fix_poly(g):
    # Make invalid polygons valid, cheaply (20 sections only)
    try:
        if g.is_valid:
            return g
        if make_valid is not None:
            return make_valid(g)
        return g.buffer(0)
    except Exception:
        return g


def load_regions(geojson_path: Path, section_field: str):
    feats = json.load(geojson_path.open())
    polys: List[Polygon] = []
    names: List[str] = []

    for f in feats.get("features", []):
        props = f.get("properties", {})
        nm = str(props.get(section_field) or props.get("name") or props.get("NAME") or "unknown")
        geom = f.get("geometry")
        if not geom:
            continue
        try:
            g = shape(geom)
        except Exception:
            continue
        if g.is_empty:
            continue
        # explode multipolygons
        if isinstance(g, (MultiPolygon,)):
            for part in g.geoms:
                polys.append(_fix_poly(part))
                names.append(nm)
        elif isinstance(g, (Polygon,)):
            polys.append(_fix_poly(g))
            names.append(nm)
        else:
            # if it's a MultiLineString/LineString boundary-only file, ignore; we need area regions
            continue

    # Build STRtree and optional prepared geoms
    tree = STRtree(polys)
    preps = [prep(p) if prep else None for p in polys]
    return polys, names, tree, preps


# ---------------------------
# Classification + sampling
# ---------------------------

def classify_point(pt: Point, cand_idxs, polys, names, preps) -> str:
    # Try fast prepared contains among candidates first
    for i in cand_idxs:
        try:
            if preps[i] is not None:
                if preps[i].contains(pt):
                    return names[i]
            else:
                if polys[i].contains(pt):
                    return names[i]
        except Exception:
            continue
    # Fallback: linear over candidates (rare)
    for i in cand_idxs:
        try:
            if polys[i].contains(pt):
                return names[i]
        except Exception:
            continue
    return ""

def nearest_province_by_shore(pt: Point, polys: List[Polygon], names: List[str]) -> tuple[str, float]:
    """Return (name, distance_km_to_nearest_shoreline) by measuring to each polygon boundary."""
    best_i = -1
    best_m = 1e12
    for i, poly in enumerate(polys):
        try:
            q = poly.boundary
            a, b = _nearest_points(pt, q)
            # a and b are on respective geometries; use their lat/lon
            d_m = haversine_m(a.y, a.x, b.y, b.x)
            if d_m < best_m:
                best_m = d_m
                best_i = i
        except Exception:
            continue
    if best_i < 0:
        return ("", 1e12)
    return names[best_i], best_m / 1000.0


def classify_point_nearshore(pt: Point, polys: List[Polygon], names: List[str], preps, shore_km: float) -> str:
    """Return province name if inside polygon OR within shore_km of nearest shoreline; else ''."""
    # exact containment first
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
    # near-shore fallback
    nm, d_km = nearest_province_by_shore(pt, polys, names)
    return nm if (nm and d_km <= shore_km) else ""

def candidate_regions_for_line(ln: LineString, tree: STRtree, polys: List[Polygon]):
    # Ask STRtree which regions intersect the line (Shapely 2.x returns indices, 1.8 returns geoms)
    try:
        idxs = tree.query(ln, predicate="intersects")  # numpy array of indices
        if getattr(idxs, "size", 0):
            return [int(i) for i in idxs]
        return []
    except TypeError:
        # Shapely 1.8 fallback: bbox candidates, then precise intersect
        geoms = tree.query(ln.envelope)
        out = []
        for i, pg in enumerate(polys):
            # quick bbox prefilter via .intersects(ln.envelope) would require pairing; iterate small N (20)
            try:
                if pg.intersects(ln):
                    out.append(i)
            except Exception:
                continue
        return out


def adaptive_samples_for_line(ln: LineString):
    # length in metres (approx: 111km per degree)
    try:
        L = max(1.0, ln.length * 111_000.0)
    except Exception:
        L = 100.0
    n = max(3, min(BASE_SAMPLES, int(L / SAMPLE_PER_METRES)))
    ts = [i/(n-1) for i in range(n)]
    return ts


def road_label(props: dict) -> str:
    ref = (props.get("ref") or "").strip()
    name = (props.get("name") or "").strip()
    hw = (props.get("highway") or props.get("route") or "").strip()
    base = ref or name or "unnamed"
    return f"{base} [{hw}]" if hw else base

def _norm(v: str) -> str:
    return (v or "").strip().lower()

YES_SET = {"yes", "designated", "permissive"}
    
def _fmt_rate(done: int, start_ts: float) -> str:
    dt = max(1e-6, time.time() - start_ts)
    return f"{done/dt:,.1f}/s"

def _print_progress(processed: int, written: int, start_ts: float):
    # Non-scrolling single-line progress for when tqdm is unavailable
    rate = _fmt_rate(processed, start_ts)
    msg = f"Processing {processed:,} feats  |  written {written:,}  |  {rate}"
    sys.stderr.write("\r" + msg[:120].ljust(120))
    sys.stderr.flush()

# ---------------------------
# Main pipeline
# ---------------------------

def process(regions_path: Path, section_field: str, roads_path: Path, out_csv: Path,
            rejects_prefix: Path | None = None, rejects_chunk: int = REJECTS_CHUNK_DEFAULT):
    polys, names, tree, preps = load_regions(regions_path, section_field)
    if len(polys) == 0:
        raise SystemExit("No polygons loaded from regions file.")

    # Show distinct region names before heavy work begins
    try:
        uniq_names = []
        seen = set()
        for n in names:
            if n not in seen:
                uniq_names.append(n)
                seen.add(n)
        sys.stderr.write("[init] Regions ({}): {}\n".format(len(uniq_names), ", ".join(uniq_names)))
        sys.stderr.flush()
    except Exception:
        pass

    out = out_csv.open("w", newline="", encoding="utf-8")
    wr = csv.writer(out)
    wr.writerow(["from_section","to_section","latitude","longitude","label"])

    # Optional rejects writer(s), split every `rejects_chunk` rows
    rej_prefix_path = rejects_prefix
    rej_file = None
    rej_wr = None
    rej_file_idx = 0
    rej_count_in_file = 0
    rej_total = 0

    def _open_new_reject_file():
        nonlocal rej_file, rej_wr, rej_file_idx, rej_count_in_file
        if rej_prefix_path is None:
            return
        # close existing
        try:
            if rej_file is not None:
                rej_file.close()
        except Exception:
            pass
        rej_file_idx += 1
        rej_count_in_file = 0
        # build path: <prefix>_rejects_<idx:06d>.csv if prefix has no extension,
        # otherwise insert _<idx:06d> before extension
        base = str(rej_prefix_path)
        if base.lower().endswith('.csv'):
            stem = base[:-4]
            path = Path(f"{stem}_{rej_file_idx:06d}.csv")
        else:
            path = Path(f"{base}_rejects_{rej_file_idx:06d}.csv")
        rej_fp = path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(rej_fp)
        writer.writerow(["type","reason","label","latitude","longitude","region_a","region_b"])
        rej_file = rej_fp
        rej_wr = writer

    def log_reject(kind: str, reason: str, label: str, geom, a: str = "", b: str = ""):
        nonlocal rej_total, rej_count_in_file
        if rej_prefix_path is None:
            return
        try:
            if (rej_wr is None) or (rej_count_in_file >= max(1, int(rejects_chunk))):
                _open_new_reject_file()
            mid = geom.interpolate(0.5, normalized=True)
            rej_wr.writerow([kind, reason, label, f"{mid.y:.7f}", f"{mid.x:.7f}", a, b])
            rej_count_in_file += 1
            rej_total += 1
        except Exception:
            pass

    written = 0
    seen_clusters: List[Tuple[str,str,float,float]] = []  # naive de-dup across entire run

    # Pre-count lines for a proper progress bar (fast on \n-delimited GeoJSONL)
    try:
        with roads_path.open("r", encoding="utf-8") as _lf:
            total_lines = sum(1 for _ in _lf)
    except Exception:
        total_lines = None

    use_tqdm = (tqdm is not None) and sys.stderr.isatty()
    pbar = None
    start_ts = time.time()
    if use_tqdm and total_lines:
        pbar = tqdm(total=total_lines, unit="feat", ncols=100, desc="Processing")

    pair_counts = defaultdict(int)  # unordered pair tally: key=(min_name, max_name)
    processed = 0

    with roads_path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
        
            # inside: for ln_no, line in enumerate(f, 1):
            processed += 1
            # progress update happens once per raw input line, even if we skip later
            if pbar is not None:
                pbar.update(1)
                if processed % 1000 == 0:
                    pbar.set_postfix(written=written)
            else:
                if processed % 2000 == 0:
                    _print_progress(processed, written, start_ts)
                    out.flush()
                    
            s = line.lstrip("\x1e").strip()
            if not s:
                continue
            try:
                feat = json.loads(s)
            except Exception:
                continue
            if feat.get("type") != "Feature":
                continue
            geom = feat.get("geometry")
            if not geom:
                continue
            if geom.get("type") not in ("LineString","MultiLineString"):
                continue
            props = feat.get("properties") or {}
            # drivable filters
            hw = (props.get("highway") or "").strip()
            rt = (props.get("route") or "").strip()

            if rt == "ferry":
                pass  # ferries handled later
            else:
                if not hw:
                    continue
                # always drop explicit non-motor ways
                if hw in NON_MOTOR:
                    continue

                ok = False
                if hw in DRIVABLE:
                    ok = True
                else:
                    # unknown highway types: require explicit motor_vehicle/motorcar permission
#                    mv = _norm(props.get("motor_vehicle"))
#                    mc = _norm(props.get("motorcar"))
#                    ok = (mv in YES_SET) or (mc in YES_SET)
                    ok = True

                if not ok:
                    continue
                    
            try:
                g = shape(geom)
            except Exception:
                continue

            def handle_ls(ls: LineString):
                nonlocal written
                # Ferry-first: classify endpoints by near-shore; accept or log reject, then return
                is_ferry = (props.get("route") or "") == "ferry"
                if is_ferry:
                    label = road_label(props)
                    try:
                        p0 = ls.interpolate(0.0, normalized=True)
                        p1 = ls.interpolate(1.0, normalized=True)
                    except Exception:
                        log_reject("ferry", "interpolate-failed", label, ls)
                        return
                    a = classify_point_nearshore(p0, polys, names, preps, SHORE_KM_FERRY)
                    b = classify_point_nearshore(p1, polys, names, preps, SHORE_KM_FERRY)
                    if a and b and a != b:
                        try:
                            mid = ls.interpolate(0.5, normalized=True)
                        except Exception:
                            log_reject("ferry", "midpoint-failed", label, ls, a, b)
                            return
                        lat, lon = mid.y, mid.x
                        s1, s2 = (a, b) if a < b else (b, a)
                        keep = True
                        for (fp,tp,clat,clon) in seen_clusters:
                            if fp==s1 and tp==s2 and haversine_m(lat,lon,clat,clon) <= CLUSTER_RADIUS_M:
                                keep = False
                                break
                        if keep:
                            wr.writerow([s1,s2,f"{lat:.7f}",f"{lon:.7f}",label])
                            seen_clusters.append((s1,s2,lat,lon))
                            written += 1
                            pair_counts[(s1, s2)] += 1
                    else:
                        log_reject("ferry", "ferry-not-crossing", label, ls, a or "", b or "")
                    return
                emitted_local = False
                # adjacency precheck
                cand_idxs = candidate_regions_for_line(ls, tree, polys)
                if len(cand_idxs) < 2:
                    # Bridge fallback: infer provinces from endpoints via near-shore classification
                    is_bridge = (props.get("bridge") or "").lower() in ("yes", "true", "1")
                    if is_bridge:
                        label = road_label(props)
                        try:
                            p0 = ls.interpolate(0.0, normalized=True)
                            p1 = ls.interpolate(1.0, normalized=True)
                        except Exception:
                            log_reject("bridge", "interpolate-failed", label, ls)
                            return
                        shore_local = SHORE_KM_BRIDGE
                        a = classify_point_nearshore(p0, polys, names, preps, shore_local)
                        b = classify_point_nearshore(p1, polys, names, preps, shore_local)
                        if a and b and a != b:
                            try:
                                mid = ls.interpolate(0.5, normalized=True)
                            except Exception:
                                log_reject("bridge", "midpoint-failed", label, ls, a, b)
                                return
                            lat, lon = mid.y, mid.x
                            s1, s2 = (a, b) if a < b else (b, a)
                            keep = True
                            for (fp,tp,clat,clon) in seen_clusters:
                                if fp==s1 and tp==s2 and haversine_m(lat,lon,clat,clon) <= CLUSTER_RADIUS_M:
                                    keep = False
                                    break
                            if keep:
                                wr.writerow([s1,s2,f"{lat:.7f}",f"{lon:.7f}",label])
                                seen_clusters.append((s1,s2,lat,lon))
                                written += 1
                                pair_counts[(s1, s2)] += 1
                                emitted_local = True
                        else:
                            log_reject("bridge", "bridge-not-crossing", label, ls, a or "", b or "")
                        return
                # sample along the line and classify
                ts = adaptive_samples_for_line(ls)
                seq = []
                for t in ts:
                    try:
                        pt = ls.interpolate(t, normalized=True)
                    except Exception:
                        continue
                    nm = classify_point(pt, cand_idxs, polys, names, preps)
                    seq.append((t, nm))
                if not seq:
                    return
                # forward/back fill
                last = ""
                for i,(t,nm) in enumerate(seq):
                    if nm: last = nm
                    else: seq[i] = (t, last)
                last = ""
                for i in range(len(seq)-1, -1, -1):
                    t,nm = seq[i]
                    if nm: last = nm
                    else: seq[i] = (t, last)
                # detect changes
                for i in range(1, len(seq)):
                    a = seq[i-1][1]; b = seq[i][1]
                    if a and b and a != b:
                        tmid = (seq[i-1][0] + seq[i][0]) / 2.0
                        try:
                            mid = ls.interpolate(tmid, normalized=True)
                        except Exception:
                            continue
                        lat, lon = mid.y, mid.x
                        s1, s2 = (a,b) if a < b else (b,a)
                        label = road_label(props)
                        # naive clustering to reduce dup spam
                        keep = True
                        for (fp,tp,clat,clon) in seen_clusters:
                            if fp==s1 and tp==s2 and haversine_m(lat,lon,clat,clon) <= CLUSTER_RADIUS_M:
                                keep = False
                                break
                        if keep:
                            wr.writerow([s1,s2,f"{lat:.7f}",f"{lon:.7f}",label])
                            seen_clusters.append((s1,s2,lat,lon))
                            written += 1
                            pair_counts[(s1, s2)] += 1
                            emitted_local = True
                # If bbox touched ≥2 regions and we did not emit a crossing, log a road reject
                if len(cand_idxs) >= 2 and not emitted_local:
                    label = road_label(props)
                    log_reject("road", "road-no-change", label, ls)

            if isinstance(g, LineString):
                handle_ls(g)
            elif isinstance(g, MultiLineString):
                for part in g.geoms:
                    handle_ls(part)


    # finish progress display
    if pbar is not None:
        pbar.close()
    else:
        _print_progress(processed, written, start_ts)
        sys.stderr.write("\n")
    # Emit unordered connection tally (e.g., Nova Scotia — New Brunswick: 7)
    try:
        if pair_counts:
            sys.stderr.write("\n[summary] Crossing counts by region pair\n")
            for (a,b), cnt in sorted(pair_counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                sys.stderr.write(f"  {a} — {b}: {cnt}\n")
            sys.stderr.flush()
    except Exception:
        pass

    try:
        if rej_file is not None:
            rej_file.close()
    except Exception:
        pass

    out.close()
    print(f"Done. Wrote {written} rows to {out_csv}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Find boundary crossings from roads/ferries GeoJSONL")
    ap.add_argument("--regions", default=DEFAULT_REGIONS)
    ap.add_argument("--section-field", default=DEFAULT_SECTION_FIELD)
    ap.add_argument("--roads", default=DEFAULT_ROADS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rejects-prefix", help="Path prefix for rejects CSVs; files split every 250k rows by default")
    ap.add_argument("--rejects-chunk", type=int, default=REJECTS_CHUNK_DEFAULT,
                    help="Max rows per rejects CSV before splitting (default 250000)")
    args = ap.parse_args()

    process(Path(args.regions), args.section_field, Path(args.roads), Path(args.out),
            rejects_prefix=Path(args.rejects_prefix) if args.rejects_prefix else None,
            rejects_chunk=args.rejects_chunk)

if __name__ == "__main__":
    main()

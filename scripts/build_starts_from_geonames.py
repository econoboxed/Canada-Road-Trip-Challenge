#!/usr/bin/env python3
import csv
from pathlib import Path

in_path = Path("CA.txt")
out_path = Path("starts.csv")

# GeoNames CA.txt columns:
# 0 geonameid
# 1 name
# 2 asciiname
# 3 alternatenames
# 4 latitude
# 5 longitude
# 6 feature class
# 7 feature code
# 8 country code
# 9 cc2
# 10 admin1 code
# 11 admin2 code
# 12 admin3 code
# 13 admin4 code
# 14 population
# 15 elevation
# 16 dem
# 17 timezone
# 18 modification date

with in_path.open("r", encoding="utf-8") as fin, \
     out_path.open("w", newline="", encoding="utf-8") as fout:

    w = csv.writer(fout)
    # travel_times_from_point.py looks for: name / lat / lon
    w.writerow(["name", "lat", "lon"])

    seen = set()

    for row in csv.reader(fin, delimiter="\t"):
        if len(row) < 19:
            continue

        name = row[1].strip()
        lat  = row[4].strip()
        lon  = row[5].strip()
        fclass = row[6].strip()
        pop_str = row[14].strip()

        # Only populated places
        if fclass != "P":
            continue

        try:
            pop = int(pop_str)
        except ValueError:
            continue

        if pop < 500:
            continue

        key = (name, lat, lon)
        if key in seen:
            continue
        seen.add(key)

        w.writerow([name, lat, lon])

print(f"Wrote filtered starts to {out_path}")

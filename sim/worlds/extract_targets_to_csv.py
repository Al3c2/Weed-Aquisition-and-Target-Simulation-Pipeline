#!/usr/bin/env python3
import re
import csv
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: extract_targets_to_csv.py <input.sdf|world> <output.csv>")
    sys.exit(1)

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

text = in_path.read_text()

pattern = re.compile(
    r"<model\s+name='([^']+)'>.*?<pose>\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*</pose>",
    re.DOTALL
)

rows = []

for m in pattern.finditer(text):
    name = m.group(1)

    # ✅ filter out posts and everything else
    if "_target_" not in name:
        continue

    x = float(m.group(2))
    y = float(m.group(3))
    z = float(m.group(4))
    rows.append((name, x, y, z))

# ✅ biggest X first
rows.sort(key=lambda r: r[1], reverse=True)

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "x", "y", "z"])
    writer.writerows(rows)

print(f"Extracted {len(rows)} targets → {out_path}")

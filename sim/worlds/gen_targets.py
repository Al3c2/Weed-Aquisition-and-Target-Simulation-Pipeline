#!/usr/bin/env python3
import argparse, random, sys, math

def fmt(n):
    return f"{n:.3f}"

p = argparse.ArgumentParser(description="Generate random bullseye targets for row1 with y-dependent visibility band")
p.add_argument("--seed", type=int, help="random seed (optional)")
p.add_argument("--out", default="-", help="output file ('-' for stdout')")
p.add_argument("--x_start", type=int, default=-51, help="start meter (inclusive), default -51")
p.add_argument("--x_end", type=int, default=-1, help="end meter (exclusive), default -1")
p.add_argument("--per_meter", type=int, default=3, help="targets per meter, default 3")
args = p.parse_args()

if args.seed is not None:
    random.seed(args.seed)
else:
    random.seed()

out = sys.stdout if args.out == "-" else open(args.out, "w")

# --- Camera / geometry assumptions from your prior math ---
# You previously had:
#   at y=0.85 -> d=0.75
#   at y=1.25 -> d=1.15
# => y_cam = y - d = 0.10
Y_CAM = 0.10

# Your prior calculations used z_cam = 0.300 in the "old" frame,
# and then world is shifted by -0.4 => z_world = z_old - 0.4
Z_CAM_OLD = 0.300
WORLD_Z_SHIFT = -0.4  # apply as: z_world = z_old + WORLD_Z_SHIFT

# Horizontal FOV (rad)
H_FOV = 0.785
TAN_HALF_FOV = math.tan(H_FOV / 2.0)

# Your row-1 y band
Y_MIN, Y_MAX = 0.9, 1.2

# --- Bullseye geometry (ultra thin) ---
PLATE_X = 0.08
PLATE_Y = 0.08
PLATE_T = 0.0005

DOT_R = 0.003
DOT_T = 0.0002
# Put dot on top of plate with a tiny gap so z-fighting doesn't happen
DOT_Z = (PLATE_T / 2.0) + (DOT_T / 2.0) + 0.00005

total = (args.x_end - args.x_start) * args.per_meter
print(f"<!-- ROW 1 BULLSEYE TARGETS: {total} targets, y-dependent z band, world z shifted by -0.4 -->", file=out)

idx = 1
for meter in range(args.x_start, args.x_end):  # intervals [meter, meter+1)
    for _ in range(args.per_meter):
        name = f"row1_target_{idx:03d}"

        x = random.uniform(meter, meter + 1.0)
        y = random.uniform(Y_MIN, Y_MAX)

        # Distance forward from camera along +y (must be > 0)
        d = y - Y_CAM
        if d <= 0.0:
            d = 1e-6

        # Visible z band in "old" frame (centered around Z_CAM_OLD)
        half_span = d * TAN_HALF_FOV
        z_min_old = Z_CAM_OLD - half_span
        z_max_old = Z_CAM_OLD + half_span

        # Shift into world coordinates
        z_min_world = z_min_old + WORLD_Z_SHIFT
        z_max_world = z_max_old + WORLD_Z_SHIFT

        # Sample z within the y-specific visible band (world coords)
        z = random.uniform(z_min_world, z_max_world)

        print(f"<model name='{name}'>", file=out)
        print("  <static>true</static>", file=out)
        print(f"  <pose>{fmt(x)} {fmt(y)} {fmt(z)} -1.5708 0 0</pose>", file=out)
        print("  <link name='link'>", file=out)

        # Plate (blue)
        print("    <collision name='plate_collision'>", file=out)
        print("      <geometry>", file=out)
        print(f"        <box><size>{PLATE_X} {PLATE_Y} {PLATE_T}</size></box>", file=out)
        print("      </geometry>", file=out)
        print("    </collision>", file=out)

        print("    <visual name='plate_visual'>", file=out)
        print("      <geometry>", file=out)
        print(f"        <box><size>{PLATE_X} {PLATE_Y} {PLATE_T}</size></box>", file=out)
        print("      </geometry>", file=out)
        print("      <material>", file=out)
        print("        <diffuse>0 0 0.9 1</diffuse>", file=out)
        print("        <ambient>0 0 0.6 1</ambient>", file=out)
        print("      </material>", file=out)
        print("    </visual>", file=out)

        # Dot (red)
        print("    <collision name='dot_collision'>", file=out)
        print(f"      <pose>0 0 {DOT_Z} 0 0 0</pose>", file=out)
        print("      <geometry>", file=out)
        print(f"        <cylinder><radius>{DOT_R}</radius><length>{DOT_T}</length></cylinder>", file=out)
        print("      </geometry>", file=out)
        print("    </collision>", file=out)

        print("    <visual name='dot_visual'>", file=out)
        print(f"      <pose>0 0 {DOT_Z} 0 0 0</pose>", file=out)
        print("      <geometry>", file=out)
        print(f"        <cylinder><radius>{DOT_R}</radius><length>{DOT_T}</length></cylinder>", file=out)
        print("      </geometry>", file=out)
        print("      <material>", file=out)
        print("        <diffuse>0.9 0 0 1</diffuse>", file=out)
        print("        <ambient>0.6 0 0 1</ambient>", file=out)
        print("      </material>", file=out)
        print("    </visual>", file=out)

        print("  </link>", file=out)
        print("</model>", file=out)

        idx += 1

print("<!-- end ROW 1 BULLSEYE TARGETS -->", file=out)
if args.out != "-":
    out.close()

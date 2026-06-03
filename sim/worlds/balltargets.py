#!/usr/bin/env python3
import argparse, random, sys, math

def fmt(n):
    return f"{n:.3f}"

# ---------------- Args ----------------
p = argparse.ArgumentParser(
    description="Generate random blue sphere targets for row1 with spacing + post/occlusion avoidance"
)
p.add_argument("--seed", type=int, help="random seed (optional)")
p.add_argument("--out", default="-", help="output file ('-' for stdout')")
p.add_argument("--x_start", type=int, default=-51, help="start meter (inclusive), default -51")
p.add_argument("--x_end", type=int, default=-1, help="end meter (exclusive), default -1")
p.add_argument("--per_meter", type=int, default=3, help="targets per meter, default 3")

# New constraints
p.add_argument("--min_dx", type=float, default=0.05, help="minimum spacing in X between any two targets (m)")
p.add_argument("--post_spacing", type=float, default=2.0, help="post spacing along X (m)")
p.add_argument("--row1_post_y", type=float, default=1.10, help="row1 post center Y (m)")
p.add_argument("--post_radius", type=float, default=0.05, help="post cylinder radius (m)")
p.add_argument("--target_radius", type=float, default=0.005, help="target sphere radius (m)")
p.add_argument("--post_clearance", type=float, default=0.03, help="extra clearance around posts (m)")
p.add_argument("--occlusion_x_band", type=float, default=0.12,
              help="if |x-post_x| < this, avoid placing target behind post in +Y (m)")
p.add_argument("--max_tries", type=int, default=200, help="max random attempts per target before fallback")
args = p.parse_args()

if args.seed is not None:
    random.seed(args.seed)
else:
    random.seed()

out = sys.stdout if args.out == "-" else open(args.out, "w")

# ---------------- Camera / geometry assumptions ----------------
Y_CAM = 0.10

Z_CAM_OLD = 0.300
WORLD_Z_SHIFT = -0.4  # z_world = z_old + WORLD_Z_SHIFT

H_FOV = 0.6
WIDTH = 848
HEIGHT = 480

TAN_HALF_HFOV = math.tan(H_FOV / 2.0)
TAN_HALF_VFOV = TAN_HALF_HFOV * (HEIGHT / WIDTH)

# Row-1 y band (your original)
Y_MIN, Y_MAX = 0.97, 1.24

# Hard caps (world frame) - your original
Z_WORLD_MIN_CAP_BEFORE = -0.2
Z_WORLD_MIN_CAP_AFTER  = -0.2
X_BUMP_THRESHOLD = -18.0
Z_WORLD_MAX_CAP = -0.1

# ---------------- Helpers ----------------
def nearest_post_x(x: float, spacing: float) -> float:
    # Posts are on a 2m grid (…, -52, -50, -48, …).
    # This snaps to the nearest multiple of spacing.
    return spacing * round(x / spacing)

def too_close_to_any_x(x: float, xs: list, min_dx: float) -> bool:
    # Global X spacing constraint
    for xx in xs:
        if abs(x - xx) < min_dx:
            return True
    return False

def collides_with_post(x: float, y: float) -> bool:
    px = nearest_post_x(x, args.post_spacing)
    py = args.row1_post_y
    # Radial collision in XY with clearance
    r_keepout = args.post_radius + args.target_radius + args.post_clearance
    dx = x - px
    dy = y - py
    return (dx*dx + dy*dy) < (r_keepout * r_keepout)

def behind_post_occluded(x: float, y: float) -> bool:
    """
    Simple occlusion heuristic:
    If you're close in X to a post, don't place the target "behind" it (higher +Y).
    """
    px = nearest_post_x(x, args.post_spacing)
    if abs(x - px) < args.occlusion_x_band:
        # If target is further +Y than post center, it sits behind/occluded by the post.
        return y > args.row1_post_y
    return False

def sample_valid_xyz(meter: int, used_xs: list):
    """
    Rejection sample x,y, then compute a z in your visible band.
    Applies:
      - min_dx between all targets in X
      - keepout around posts
      - avoid behind-post occlusion near posts
    """
    for attempt in range(args.max_tries):
        x = random.uniform(meter, meter + 1.0)

        # Enforce global X spacing (this fixes your "same x" complaint)
        if too_close_to_any_x(x, used_xs, args.min_dx):
            continue

        y = random.uniform(Y_MIN, Y_MAX)

        # Avoid collisions with the post cylinder
        if collides_with_post(x, y):
            continue

        # Avoid “behind post” placements that are typically occluded / unreachable
        if behind_post_occluded(x, y):
            continue

        # ---- Your original Z logic ----
        d = y - Y_CAM
        if d <= 0.0:
            d = 1e-6

        half_span = d * TAN_HALF_VFOV
        z_min_old = Z_CAM_OLD - half_span
        z_max_old = Z_CAM_OLD + half_span

        z_min_world = z_min_old + WORLD_Z_SHIFT
        z_max_world = z_max_old + WORLD_Z_SHIFT

        z_min_cap = Z_WORLD_MIN_CAP_AFTER if x > X_BUMP_THRESHOLD else Z_WORLD_MIN_CAP_BEFORE

        z_min_world = max(z_min_world, z_min_cap)
        z_max_world = min(z_max_world, Z_WORLD_MAX_CAP)

        if z_max_world < z_min_world:
            z_min_world = z_max_world = min(max(z_min_world, z_min_cap), Z_WORLD_MAX_CAP)

        z = random.uniform(z_min_world, z_max_world)

        return x, y, z

    # Fallback: if we fail too often, relax ONLY the behind-post rule slightly by biasing y to front side
    # (Still keeps collision + min_dx!)
    for attempt in range(args.max_tries):
        x = random.uniform(meter, meter + 1.0)
        if too_close_to_any_x(x, used_xs, args.min_dx):
            continue

        # Bias y to be in front of post center, not behind
        y = random.uniform(Y_MIN, min(Y_MAX, args.row1_post_y - (args.post_radius + args.post_clearance)))

        if collides_with_post(x, y):
            continue

        d = y - Y_CAM
        if d <= 0.0:
            d = 1e-6

        half_span = d * TAN_HALF_VFOV
        z_min_old = Z_CAM_OLD - half_span
        z_max_old = Z_CAM_OLD + half_span

        z_min_world = z_min_old + WORLD_Z_SHIFT
        z_max_world = z_max_old + WORLD_Z_SHIFT

        z_min_cap = Z_WORLD_MIN_CAP_AFTER if x > X_BUMP_THRESHOLD else Z_WORLD_MIN_CAP_BEFORE

        z_min_world = max(z_min_world, z_min_cap)
        z_max_world = min(z_max_world, Z_WORLD_MAX_CAP)

        if z_max_world < z_min_world:
            z_min_world = z_max_world = min(max(z_min_world, z_min_cap), Z_WORLD_MAX_CAP)

        z = random.uniform(z_min_world, z_max_world)
        return x, y, z

    return None

# ---------------- Generate ----------------
total = (args.x_end - args.x_start) * args.per_meter
print(
    f"<!-- ROW 1 BLUE TARGETS: {total} spheres (r={args.target_radius:.3f}), "
    f"min_dx={args.min_dx:.3f}m, post_keepout={(args.post_radius+args.target_radius+args.post_clearance):.3f}m, "
    f"occlusion_x_band={args.occlusion_x_band:.3f}m -->",
    file=out,
)

idx = 1
used_xs = []

for meter in range(args.x_start, args.x_end):
    for _ in range(args.per_meter):
        name = f"row1_target_{idx:03d}"

        xyz = sample_valid_xyz(meter, used_xs)
        if xyz is None:
            print(f"<!-- WARNING: Could not place target {name} in meter [{meter},{meter+1}) under constraints -->", file=out)
            idx += 1
            continue

        x, y, z = xyz
        used_xs.append(x)

        print(f"<model name='{name}'>", file=out)
        print("  <static>true</static>", file=out)
        print(f"  <pose>{fmt(x)} {fmt(y)} {fmt(z)} 0 0 0</pose>", file=out)
        print("  <link name='link'>", file=out)
        print("    <collision name='collision'>", file=out)
        print(f"      <geometry><sphere><radius>{args.target_radius:.3f}</radius></sphere></geometry>", file=out)
        print("    </collision>", file=out)
        print("    <visual name='visual'>", file=out)
        print(f"      <geometry><sphere><radius>{args.target_radius:.3f}</radius></sphere></geometry>", file=out)
        print("      <material>", file=out)
        print("        <diffuse>0 0 0.9 1</diffuse>", file=out)
        print("        <ambient>0 0 0.6 1</ambient>", file=out)
        print("      </material>", file=out)
        print("    </visual>", file=out)
        print("  </link>", file=out)
        print("</model>", file=out)

        idx += 1

print("<!-- end ROW 1 BLUE TARGETS -->", file=out)

if args.out != "-":
    out.close()

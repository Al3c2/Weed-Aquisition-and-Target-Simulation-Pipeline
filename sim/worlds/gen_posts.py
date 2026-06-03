#!/usr/bin/env python3

def post_model_xml(name: str, x: float, y: float, z: float) -> str:
    return f"""\
    <model name='{name}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name='vine_link'>
        <collision name='post_collision'>
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>1.2</length>
            </cylinder>
          </geometry>
          <pose>0 0 0.6 0 0 0</pose>
        </collision>
        <visual name='post_visual'>
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>1.2</length>
            </cylinder>
          </geometry>
          <pose>0 0 0.6 0 0 0</pose>
          <material>
            <diffuse>0.4 0.3 0.2 1</diffuse>
            <ambient>0.4 0.3 0.2 1</ambient>
          </material>
        </visual>

        <collision name='ball_collision'>
          <geometry>
            <sphere>
              <radius>0.07</radius>
            </sphere>
          </geometry>
          <pose>0 0 1.0 0 0 0</pose>
        </collision>
        <visual name='ball_visual'>
          <geometry>
            <sphere>
              <radius>0.07</radius>
            </sphere>
          </geometry>
          <pose>0 0 1.0 0 0 0</pose>
          <material>
            <diffuse>0.9 0.1 0.1 1</diffuse>
            <ambient>0.9 0.1 0.1 1</ambient>
          </material>
        </visual>
      </link>
    </model>
"""

def frange(start: float, end: float, step: float):
    x = start
    # include end if it lands exactly on it (within float tolerance)
    while x <= end + 1e-9:
        yield x
        x += step

def main():
    # ====== CUSTOMIZE THESE ======
    x_start = -100
    x_end   = 0
    spacing = 2.0      # posts every 2m (change to what you want)

    y_row1 =  1.1
    y_row2 = -1.1
    z_spawn = -0.5     # your model pose z
    # ============================

    print("    <!-- =========== GENERATED VINE POSTS (2 ROWS) =========== -->")
    print(f"    <!-- X from {x_start} to {x_end} with spacing {spacing} -->")
    print(f"    <!-- Row1 Y={y_row1}, Row2 Y={y_row2}, Z={z_spawn} -->\n")

    i = 0
    for x in frange(x_start, x_end, spacing):
        name1 = f"row1_post_{i:03d}"
        name2 = f"row2_post_{i:03d}"
        print(post_model_xml(name1, x, y_row1, z_spawn))
        print(post_model_xml(name2, x, y_row2, z_spawn))
        i += 1

if __name__ == "__main__":
    main()

sr/bin/env python3
"""
UR3 Beam Pointing Workspace Tester

Tests reachability of target positions and provides recommendations.
Spawns the UR3 at: [0, 0.2, 0.35]

Usage:
    python3 workspace_tester.py --test-grid
    python3 workspace_tester.py --test-single 0.1 0.8 0.15
    python3 workspace_tester.py --generate-targets 50
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
import argparse
import time
import numpy as np


class WorkspaceTester(Node):
    def __init__(self):
        super().__init__('workspace_tester')
        
        self.target_pub = self.create_publisher(PoseStamped, 'target_pose', 10)
        
        # Wait for node to be ready
        time.sleep(1.0)
        self.get_logger().info('🧪 Workspace Tester ready')

    def test_single_point(self, x, y, z):
        """Test a single target point"""
        msg = PoseStamped()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        
        self.get_logger().info(f'📍 Testing point [{x:.2f}, {y:.2f}, {z:.2f}]')
        self.target_pub.publish(msg)
        time.sleep(0.5)

    def test_grid(self, x_range, y_range, z_range, step=0.1):
        """Test a grid of points"""
        self.get_logger().info(f'🔬 Testing grid: X{x_range} Y{y_range} Z{z_range} step={step}')
        
        points_tested = 0
        for x in np.arange(x_range[0], x_range[1] + step, step):
            for y in np.arange(y_range[0], y_range[1] + step, step):
                for z in np.arange(z_range[0], z_range[1] + step, step):
                    self.test_single_point(x, y, z)
                    points_tested += 1
                    time.sleep(3.0)  # Give time for execution
        
        self.get_logger().info(f'✅ Tested {points_tested} points')

    def generate_vineyard_targets(self, num_rows=5, weeds_per_row=10):
        """Generate realistic vineyard weed targets"""
        self.get_logger().info(f'🌿 Generating {num_rows} vineyard rows, {weeds_per_row} weeds/row')
        
        targets = []
        for row in range(num_rows):
            # Row spacing: 0.5m to 1.2m from robot
            y_base = 0.5 + row * 0.15
            
            for weed in range(weeds_per_row):
                # Weeds distributed along row width
                x = np.random.uniform(-0.2, 0.2)
                y = y_base + np.random.uniform(-0.05, 0.05)  # slight variation
                z = np.random.uniform(0.05, 0.25)  # weed height
                
                targets.append((x, y, z))
        
        # Sort by Y (front to back)
        targets.sort(key=lambda p: p[1])
        
        self.get_logger().info(f'📋 Generated {len(targets)} targets')
        
        for i, (x, y, z) in enumerate(targets):
            self.get_logger().info(f'  Target {i+1}: [{x:.3f}, {y:.3f}, {z:.3f}]')
            self.test_single_point(x, y, z)
            time.sleep(2.5)  # Time between targets

    def call_workspace_analysis(self):
        """Call the workspace analysis service"""
        client = self.create_client(Trigger, 'analyze_workspace')
        
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('❌ Workspace analysis service not available')
            return
        
        self.get_logger().info('🔍 Requesting workspace analysis...')
        request = Trigger.Request()
        future = client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'\n{response.message}')
        else:
            self.get_logger().error('❌ Service call failed')


def main():
    parser = argparse.ArgumentParser(description='Test UR3 beam pointing workspace')
    parser.add_argument('--test-grid', action='store_true',
                       help='Test a grid of points')
    parser.add_argument('--test-single', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Test a single point')
    parser.add_argument('--generate-targets', type=int, metavar='NUM_WEEDS',
                       help='Generate N vineyard weed targets')
    parser.add_argument('--analyze', action='store_true',
                       help='Run workspace analysis service')
    
    # Grid parameters
    parser.add_argument('--x-range', nargs=2, type=float, default=[-0.25, 0.25],
                       metavar=('MIN', 'MAX'), help='X range (default: -0.25 0.25)')
    parser.add_argument('--y-range', nargs=2, type=float, default=[0.5, 1.2],
                       metavar=('MIN', 'MAX'), help='Y range (default: 0.5 1.2)')
    parser.add_argument('--z-range', nargs=2, type=float, default=[0.05, 0.25],
                       metavar=('MIN', 'MAX'), help='Z range (default: 0.05 0.25)')
    parser.add_argument('--step', type=float, default=0.15,
                       help='Grid step size (default: 0.15)')
    
    args = parser.parse_args()
    
    rclpy.init()
    tester = WorkspaceTester()
    
    try:
        if args.analyze:
            tester.call_workspace_analysis()
        elif args.test_single:
            x, y, z = args.test_single
            tester.test_single_point(x, y, z)
        elif args.test_grid:
            tester.test_grid(args.x_range, args.y_range, args.z_range, args.step)
        elif args.generate_targets:
            weeds_per_row = max(1, args.generate_targets // 5)
            tester.generate_vineyard_targets(num_rows=5, weeds_per_row=weeds_per_row)
        else:
            print("\n🎯 UR3 Beam Pointing Workspace Information")
            print("=" * 60)
            print(f"UR3 base position: [0.0, 0.2, 0.35]")
            print(f"Beam length: 0.65m (searchable: 0.35-1.20m)")
            print()
            print("Recommended target ranges (world frame):")
            print(f"  X: -0.25 to +0.25 m  (lateral, across vineyard row)")
            print(f"  Y: +0.50 to +1.20 m  (forward, along Swincar movement)")
            print(f"  Z: +0.00 to +0.30 m  (height, weed elevation)")
            print()
            print("Example commands:")
            print("  Test single point:     --test-single 0.1 0.8 0.15")
            print("  Test grid:             --test-grid --step 0.2")
            print("  Generate weeds:        --generate-targets 50")
            print("  Analyze workspace:     --analyze")
            print()
            
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

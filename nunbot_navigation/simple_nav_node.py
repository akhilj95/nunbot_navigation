import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import tf2_ros
import math
import time

class SimpleNavNode(Node):
    def __init__(self):
        super().__init__('simple_nav_node')

        # Parameters and state
        self.initial_pose = [0.0, 0.0, 0.0]  # Set initial known pose coords (point A)
        self.waypoints = [
            [0.1, 0.1],   # Start
            [-0.5, -0.5],   # Define B, C, D coords here
            [0, 0]    # Return to A
        ]
        self.current_waypoint_idx = 1
        self.amcl_pose = None
        self.button_on = False
        self.amcl_initialized = False

        # Subscribers
        self.create_subscription(Bool, '/nunbot/button_onoff', self.button_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Timer for main loop
        self.timer = self.create_timer(0.5, self.main_loop)

        # Costmap data placeholder
        self.latest_costmap = None

        self.get_logger().info('Simple Navigation Node initialized.')

    def button_callback(self, msg):
        if msg.data and not self.button_on:
            self.button_on = True
            self.get_logger().info('Button ON detected, starting navigation...')
            self.send_initial_pose()
        elif not msg.data and self.button_on:
            self.button_on = False
            self.get_logger().info('Button OFF detected, stopping navigation.')

    def send_initial_pose(self):
        # Publish initial pose to AMCL
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.pose.pose.position.x = self.initial_pose[0]
        initial_pose_msg.pose.pose.position.y = self.initial_pose[1]
        # Orientation from theta angle (yaw)
        import tf_transformations
        q = tf_transformations.quaternion_from_euler(0, 0, self.initial_pose[2])
        initial_pose_msg.pose.pose.orientation.x = q[0]
        initial_pose_msg.pose.pose.orientation.y = q[1]
        initial_pose_msg.pose.pose.orientation.z = q[2]
        initial_pose_msg.pose.pose.orientation.w = q[3]
        # Covariance small for certainty
        initial_pose_msg.pose.covariance = [0.0]*36
        self.initial_pose_pub.publish(initial_pose_msg)
        self.get_logger().info('Initial pose published. Waiting 10 seconds for AMCL stabilization.')
        self.amcl_initialized = False  # Reset
        # Wait here 10 seconds
        time.sleep(10.0)
        
    def amcl_pose_callback(self, msg):
        self.amcl_pose = msg.pose.pose
        # Consider AMCL initialized if pose received
        if not self.amcl_initialized:
            self.amcl_initialized = True
            
        self.get_logger().info('AMCL pose data received.')

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def main_loop(self):
        if not self.button_on or not self.amcl_initialized or self.latest_costmap is None:
            # Wait for start, AMCL init, and costmap readiness
            return

        # Check obstacle in a 30 degree cone and 1m range
        if self.is_obstacle_in_cone():
            self.get_logger().info('Obstacle detected in path. Robot stopped.')
            self.publish_twist(0.0, 0.0)
            return

        # Get current pose - using latest amcl_pose
        if self.amcl_pose is None:
            self.get_logger().warning('No AMCL pose available yet.')
            return

        robot_x = self.amcl_pose.position.x
        robot_y = self.amcl_pose.position.y
        robot_yaw = self.get_yaw_from_quaternion(self.amcl_pose.orientation)

        # Target waypoint coords
        target_x, target_y = self.waypoints[self.current_waypoint_idx]

        # Check distance to target
        dist = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
        if dist < 0.15:
            # Reached waypoint, stop and wait
            self.get_logger().info(f'Waypoint {self.current_waypoint_idx} reached. Waiting for 200 seconds.')
            self.publish_twist(0.0, 0.0)
            time.sleep(200.0)
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.get_logger().info('All waypoints visited. Navigation complete.')
                self.button_on = False
            return

        # Compute direction angle to waypoint
        target_angle = math.atan2(target_y - robot_y, target_x - robot_x)
        angle_diff = self.angle_diff(target_angle, robot_yaw)

        # For simplicity, only move linearly straight (ignore rotation commands)
        linear_speed = 0.17
        self.publish_twist(linear_speed, 0.0)

        # Move for 1 second then stop
        time.sleep(1.0)
        self.publish_twist(0.0, 0.0)

    def publish_twist(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def is_obstacle_in_cone(self):
        # Process latest_costmap to detect obstacles in 30 deg cone, 1m range in front
        # Assuming costmap is an OccupancyGrid with info: resolution, width, height, origin
        costmap = self.latest_costmap
        if costmap is None:
            return False

        # Parameters for cone
        cone_angle_rad = math.radians(30)
        max_range = 1.0

        resolution = costmap.info.resolution
        width = costmap.info.width
        height = costmap.info.height

        # Costmap origin pose (map frame)
        origin_x = costmap.info.origin.position.x
        origin_y = costmap.info.origin.position.y

        data = costmap.data

        # Check cells in front cone of robot (simplified: local frame origin at robot)
        # We'll check a semicircle area in front of the robot transformed to the costmap grid
        # Since costmap is centered differently, calculation adapted as per costmap definition

        # Check 1m radius in costmap cells
        max_cells = int(max_range / resolution)

        # Iterate over relevant cells within a square bounding the circle
        # For this example, check cells in front half (x > 0) within cone angle
        # Costmap origin is map frame, but we need robot frame points converted to map frame or vice versa.
        # Simplified: Assume robot at center of costmap, which might not always be true - adjust accordingly.

        # Convert robot map coordinates to costmap indices
        # Robot is at origin in local frame, get robot map pose from AMCL
        robot_pose = self.amcl_pose
        if robot_pose is None:
            return False

        robot_map_x = robot_pose.position.x
        robot_map_y = robot_pose.position.y
        robot_yaw = self.get_yaw_from_quaternion(robot_pose.orientation)

        # Scan costmap cells in bounding box with offset by costmap origin
        for i in range(width):
            for j in range(height):
                # Cell center coordinates in map frame
                cell_x = origin_x + (i + 0.5) * resolution
                cell_y = origin_y + (j + 0.5) * resolution

                # Vector from robot to cell
                dx = cell_x - robot_map_x
                dy = cell_y - robot_map_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > max_range or dist == 0.0:
                    continue

                # Angle in robot frame, normalize between -pi, pi
                angle = math.atan2(dy, dx) - robot_yaw
                angle = (angle + math.pi) % (2 * math.pi) - math.pi

                if abs(angle) <= cone_angle_rad / 2.0:
                    # Check costmap cell occupancy > threshold (e.g. 50)
                    idx = j * width + i
                    if data[idx] > 50:
                        return True

        return False

    def get_yaw_from_quaternion(self, q):
        # Convert quaternion to yaw angle
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return euler[2]

    def angle_diff(self, a, b):
        # Calculate shortest angle difference between a and b in radians
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff


def main(args=None):
    rclpy.init(args=args)
    node = SimpleNavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

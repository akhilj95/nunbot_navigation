#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import math
import time
import tf_transformations
from enum import Enum
import numpy as np

# ROS2 message imports
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

class NavigationState(Enum):
    WAITING_FOR_START = 1
    INITIALIZING_AMCL = 2
    WAITING_FOR_AMCL = 3
    NAVIGATING = 4
    AT_WAYPOINT = 5
    COMPLETED = 6
    STOPPED = 7

class SimpleWaypointNavigator(Node):
    def __init__(self):
        super().__init__('simple_waypoint_navigator')

        # Parameters and state
        self.initial_pose = [0.0, 0.0, 0.0]  # x,y, theta
        
        self.waypoints = [
            [0.75, 0.5],
            [0.1, 0.2],
            [-0.3,0.2]
        ]

        """  
        self.waypoints = [
            [0.75, 0.2],
            [1.48, 0.1],   # [2.48, 0.1]
            [4.6, 1.2],   # Define B, C, D coords here
            [7, 2.3]
        ]
        """
        self.current_waypoint_idx = 0
        self.forward = True  # for traversing waypoints list in forward/backward 

        # Navigation parameters
        self.linear_speed = 0.16  # m/s
        self.rotation_speed = 0.22  # m/s
        self.move_duration = 1.0  # seconds
        self.waypoint_tolerance = 0.1  # meters
        self.obstacle_check_distance = 1.0  # meters
        self.obstacle_check_angle = 30.0  # degrees (total cone)
        self.waypoint_pause_duration = 10.0  # seconds

        # State management
        self.state = NavigationState.WAITING_FOR_START
        self.button_on = False
        self.amcl_initialized = False
        self.amcl_pose = None
        self.last_move_time = None 
        self.waypoint_reached_time = None

        # Subscribers
        self.create_subscription(Bool, '/nunbot/button_onoff', self.button_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_callback, 5)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Timer for main loop
        self.timer = self.create_timer(0.1, self.navigation_loop)

        # Costmap data placeholder
        self.latest_costmap = None

        self.get_logger().info('Simple Navigation Node initialized.')
        self.get_logger().info(f'Waypoints: {self.waypoints}')

    def button_callback(self, msg):
        """Handle button press to start navigation"""
        if msg.data and self.state == NavigationState.WAITING_FOR_START:
            self.button_on = True
            self.get_logger().info('Button pressed! Starting navigation...')
            self.state = NavigationState.INITIALIZING_AMCL
            self.send_initial_pose()
        elif not msg.data and self.button_on:
            self.button_on = False
            self.state = NavigationState.WAITING_FOR_START
            self.current_waypoint_idx = 0
            self.get_logger().info('Button OFF detected, stopping navigation.')

    def send_initial_pose(self):
        # Publish initial pose to AMCL
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.pose.pose.position.x = self.initial_pose[0]
        initial_pose_msg.pose.pose.position.y = self.initial_pose[1]

        # Orientation from theta angle (yaw)
        q = tf_transformations.quaternion_from_euler(0, 0, self.initial_pose[2])
        initial_pose_msg.pose.pose.orientation.x = q[0]
        initial_pose_msg.pose.pose.orientation.y = q[1]
        initial_pose_msg.pose.pose.orientation.z = q[2]
        initial_pose_msg.pose.pose.orientation.w = q[3]
        # Covariance small for certainty
        # initial_pose_msg.pose.covariance = [0.01]*36
        initial_pose_msg.pose.covariance = [
            0.005, 0,     0,     0,     0,     0,    # variance x
            0,     0.005, 0,     0,     0,     0,    # variance y
            0,     0,     1e6,   0,     0,     0,    # z variance very high (unknown)
            0,     0,     0,     1e6,   0,     0,    # roll variance very high (unknown)
            0,     0,     0,     0,     1e6,   0,    # pitch variance very high (unknown)
            0,     0,     0,     0,     0,     0.05 # yaw variance
        ]
        self.initial_pose_pub.publish(initial_pose_msg)
        self.get_logger().info('Initial pose published. Waiting 5 seconds for AMCL stabilization.')

        # Wait for 5 seconds as specified
        self.initialization_start_time = self.get_clock().now()
        self.state = NavigationState.WAITING_FOR_AMCL

        self.amcl_initialized = False  # Reset till next message

        # Wait here 5 seconds
        time.sleep(5.0)
                   
    def send_pose(self,pose):
        # Publish initial pose to AMCL
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose = pose

        # Covariance small for certainty
        pose_msg.pose.covariance = [0.01]*36

        self.initial_pose_pub.publish(pose_msg)
        self.get_logger().info('Last known pose published. Waiting 5 seconds for AMCL stabilization.')

        # Wait for 5 seconds as specified
        self.initialization_start_time = self.get_clock().now()
        self.state = NavigationState.WAITING_FOR_AMCL

        self.amcl_initialized = False  # Reset till next message

        # Wait here 5 seconds
        time.sleep(5.0)

    def amcl_pose_callback(self, msg):
        self.amcl_pose = msg.pose.pose
        # Consider AMCL initialized if pose received
        if not self.amcl_initialized:
            self.amcl_initialized = True
            self.get_logger().info('First AMCL pose data received.')

    def costmap_callback(self, msg):
        self.latest_costmap = msg


    def get_yaw_from_quaternion(self, q):
        # Convert quaternion to yaw angle
        euler = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return euler[2]

    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def calculate_direction_to_waypoint(self, current_pos, target_waypoint):
        """Calculate the direction (angle) to the next waypoint"""
        dx = target_waypoint[0] - current_pos[0]
        dy = target_waypoint[1] - current_pos[1]
        return math.atan2(dy, dx)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return angle

    def send_twist_command(self, linear_vel, angular_vel, direction, duration):
        """Send twist command for specified duration"""
        twist = Twist()
        twist.linear.x = linear_vel * math.cos(direction)
        twist.linear.y = linear_vel * math.sin(direction)
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)

        # Store time when command was sent
        self.last_move_time = self.get_clock().now()
        self.move_duration_remaining = duration
    
    def stop_robot(self):
        """Send stop command"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def navigation_loop(self):
        """Main navigation loop called at 10Hz"""
        current_time = self.get_clock().now()

        if self.state == NavigationState.WAITING_FOR_START:
            # Do nothing, wait for button press
            return

        elif self.state == NavigationState.WAITING_FOR_AMCL:
            # Wait for 10 seconds, then check if AMCL is initialized
            elapsed = (current_time - self.initialization_start_time).nanoseconds / 1e9

            if elapsed > 10.0:  # 10 seconds elapsed
                if self.amcl_initialized:
                    self.get_logger().info('AMCL initialized successfully!')
                    self.state = NavigationState.NAVIGATING
                else:
                    self.get_logger().error('AMCL failed to initialize properly. Retrying')
                    self.send_initial_pose()

        elif self.state == NavigationState.NAVIGATING:
            # Check if we need to stop previous movement
            if (self.last_move_time is not None and 
                hasattr(self, 'move_duration_remaining')):
                elapsed = (current_time - self.last_move_time).nanoseconds / 1e9
                if elapsed >= self.move_duration_remaining:
                    self.stop_robot()
                    self.last_move_time = None

            # Only proceed if we're not currently moving
            if self.last_move_time is None:
                self.execute_navigation_step()

        elif self.state == NavigationState.AT_WAYPOINT:
            # Wait at waypoint for specified duration
            if self.waypoint_reached_time is not None:
                elapsed = (current_time - self.waypoint_reached_time).nanoseconds / 1e9
                if elapsed >= self.waypoint_pause_duration:
                    self.get_logger().info('Waypoint pause completed, continuing...')

                    if self.forward:
                        self.current_waypoint_idx += 1
                        if self.current_waypoint_idx >= len(self.waypoints) - 1:
                            self.forward = False
                    else:
                        self.current_waypoint_idx -= 1
                        if self.current_waypoint_idx <= 0:
                            self.forward = True
                    
                    self.state = NavigationState.NAVIGATING
                    self.waypoint_reached_time = None

        elif self.state == NavigationState.COMPLETED:
            # Navigation completed, do nothing
            pass

    def is_obstacle_in_box(self):
        # Assuming costmap is an OccupancyGrid with info: resolution, width, height, origin
        costmap = self.latest_costmap
        if costmap is None:
            self.get_logger().warning('Cannot get costmap')
            return True
        
        resolution = costmap.info.resolution
        width = costmap.info.width
        height = costmap.info.height

        data = costmap.data

        # Convert costmap data (usually a flat list) into a numpy array with shape (height, width)
        costmap_array = np.array(data).reshape((height, width))

        # Check how many cells have values greater than 50
        count_high_prob = np.sum(costmap_array > 50)
        
        # Check if atleast 10 cells have these
        if count_high_prob >= 10:
            return True
        else:
            return False

    def execute_navigation_step(self):
        """Execute one step of the navigation process"""
        # Get current pose - using latest amcl_pose
        if self.amcl_pose is None:
            self.get_logger().warning('Cannot get current pose, skipping navigation step')
            return

        current_x = self.amcl_pose.position.x
        current_y = self.amcl_pose.position.y
        current_yaw = self.get_yaw_from_quaternion(self.amcl_pose.orientation)

        target_waypoint = self.waypoints[self.current_waypoint_idx]

        # Check if we've reached the current waypoint
        distance_to_waypoint = self.calculate_distance(
            (current_x, current_y), target_waypoint)

        if distance_to_waypoint <= self.waypoint_tolerance:
            self.get_logger().info(
                f'Reached waypoint {self.current_waypoint_idx}: {target_waypoint}')
            self.state = NavigationState.AT_WAYPOINT
            self.waypoint_reached_time = self.get_clock().now()
            return

        # Check for obstacles
        if self.is_obstacle_in_box():
            self.get_logger().warning('Obstacle detected! Stopping.')
            self.stop_robot()
            return

        # Calculate direction to waypoint
        target_direction = self.calculate_direction_to_waypoint(
            (current_x, current_y), target_waypoint)

        # Calculate angular error
        angular_error = self.normalize_angle(target_direction - current_yaw)

        # If we need to turn significantly, turn first
        if abs(angular_error) > math.radians(10):  # 10 degrees threshold
            angular_vel = self.rotation_speed if angular_error > 0 else -self.rotation_speed
            self.get_logger().info(f'Turning towards waypoint, angular error: {math.degrees(angular_error):.1f} degrees')
            self.send_twist_command(0.0, angular_vel, 0.0, 0.2)  # Turn for 0.5 seconds
        else:
            # Move forward towards waypoint
            self.get_logger().info(f'Moving towards waypoint {self.current_waypoint_idx}: {target_waypoint}')
            self.send_twist_command(self.linear_speed, 0.0, angular_error, self.move_duration) 

def main(args=None):
    rclpy.init(args=args)

    navigator = SimpleWaypointNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
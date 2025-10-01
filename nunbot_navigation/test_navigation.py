import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class MapPrinter(Node):
    def __init__(self):
        super().__init__('map_printer')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',  # Adjust topic as needed
            self.map_callback,
            10)
        self.map_received = False

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Define characters for display
        # # for occupied, . for free, ? for unknown
        def cell_char(value):
            if value == -1:
                return '?'
            elif 0 < value <= 30:
                return '0'
            elif 30 < value <= 50:
                return '1'
            elif 50 < value <= 75:
                return '*'
            elif value > 75:
                return '#'
            else:
                return '/'

        print("Occupancy Grid:")
        for row in data:
            print(''.join(cell_char(cell) for cell in row))

        self.map_received = True

def main(args=None):
    rclpy.init(args=args)
    node = MapPrinter()
    while rclpy.ok() and not node.map_received:
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
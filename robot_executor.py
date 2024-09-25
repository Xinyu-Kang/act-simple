import rospy
from std_msgs.msg import Float64MultiArray
import threading
import numpy as np

class RobotExecutor:
    def __init__(self, control_rate=20):
        """
        Initializes the RobotExecutor.

        Args:
            control_rate (int): Frequency (in Hz) at which to send control commands.
        """
        # Initialize ROS node
        rospy.init_node('robot_executor', anonymous=True)

        # Publisher to send commands to the robot
        self.command_pub = rospy.Publisher('/robot/command', Float64MultiArray, queue_size=10)

        # Subscriber to receive the predicted next target positions
        rospy.Subscriber('/robot/next_target', Float64MultiArray, self.target_callback)

        # Latest target position
        self.latest_target = None

        # Lock for thread-safe access to the latest target
        self.target_lock = threading.Lock()

        # Control loop rate
        self.control_rate = rospy.Rate(control_rate)

        # Start the control loop in a separate thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def target_callback(self, msg):
        """
        Callback function to handle new target positions.

        Args:
            msg (Float64MultiArray): The latest target position.
        """
        with self.target_lock:
            self.latest_target = np.array(msg.data)
            rospy.loginfo(f"Received new target: {self.latest_target}")

    def control_loop(self):
        """
        Control loop that continuously moves the robot towards the latest target position.
        """
        while not rospy.is_shutdown():
            # Get the latest target position
            with self.target_lock:
                target = self.latest_target

            if target is not None:
                # Compute the control command based on the target
                command = self.compute_control_command(target)

                # Publish the command
                command_msg = Float64MultiArray()
                command_msg.data = command.tolist()
                self.command_pub.publish(command_msg)
                rospy.loginfo(f"Published control command: {command}")

            # Sleep to maintain the control rate
            self.control_rate.sleep()

    def compute_control_command(self, target):
        """
        Computes the control command to move the robot towards the target.

        Args:
            target (np.array): The latest target position [x, y, z].

        Returns:
            np.array: The control command to be sent to the robot.
        """
        # Placeholder for robot's current position
        # In a real implementation, you should get the robot's current position from sensors or state estimation
        current_position = self.get_current_robot_position()

        # Compute the direction vector towards the target
        direction = target - current_position

        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.zeros_like(direction)

        # Set the speed (you can adjust the speed as needed)
        speed = 0.05  # units per second

        # Compute the incremental movement
        increment = direction * speed / self.control_rate.sleep_dur.to_sec()

        # Compute the new position command
        new_position = current_position + increment

        # For this example, we'll send the new position as the command
        # In practice, you may need to convert this to joint commands or velocities
        command = new_position

        # Update current_position for the next iteration
        self.current_position = new_position

        return command

    def get_current_robot_position(self):
        """
        Gets the robot's current position.

        Returns:
            np.array: The current position [x, y, z] of the robot's end effector.
        """
        # Placeholder implementation
        # In a real implementation, you should get the robot's current position from sensors or state estimation
        if not hasattr(self, 'current_position'):
            self.current_position = np.array([0.0, 0.0, 0.0])
        return self.current_position

if __name__ == '__main__':
    try:
        executor = RobotExecutor(control_rate=20)  # Control loop at 20 Hz
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
from cv_bridge import CvBridge
import numpy as np
import h5py
import os

class DataCollector:
    def __init__(self, dataset_path):
        self.bridge = CvBridge()
        self.dataset_path = dataset_path
        self.hdf5_file = h5py.File(self.dataset_path, 'w')

        # Create groups within the HDF5 file
        self.images_group = self.hdf5_file.create_group("images")
        self.status_group = self.hdf5_file.create_group("status")
        self.actions_group = self.hdf5_file.create_group("actions")

        # Index for dataset entries
        self.index = 0

        # Subscribers using message_filters
        image_sub = Subscriber("/camera/image_raw", Image)
        status_sub = Subscriber("/robot/status", Float64MultiArray)
        action_sub = Subscriber("/robot/next_target", Float64MultiArray)

        # Approximate Time Synchronizer
        ats = ApproximateTimeSynchronizer(
            [image_sub, status_sub, action_sub],
            queue_size=10,
            slop=0.1
        )
        ats.registerCallback(self.synced_callback)

    def synced_callback(self, image_msg, status_msg, action_msg):
        try:
            # Process image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            self.latest_image_timestamp = image_msg.header.stamp.to_sec()

            # Process robot status
            self.latest_status = np.array(status_msg.data)
            self.latest_status_timestamp = status_msg.header.stamp.to_sec()

            # Process action
            self.latest_action = np.array(action_msg.data)
            self.latest_action_timestamp = action_msg.header.stamp.to_sec()

            # Save data
            self.save_data()

        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

    def save_data(self):
        # Save Image
        img_encoded = cv2.imencode('.jpg', self.latest_image)[1].tobytes()
        img_ds = self.images_group.create_dataset(
            f"image_{self.index}",
            data=np.void(img_encoded),
            compression="gzip"
        )
        img_ds.attrs['timestamp'] = self.latest_image_timestamp

        # Save Robot Status
        status_ds = self.status_group.create_dataset(
            f"status_{self.index}",
            data=self.latest_status
        )
        status_ds.attrs['timestamp'] = self.latest_status_timestamp

        # Save Action
        action_ds = self.actions_group.create_dataset(
            f"action_{self.index}",
            data=self.latest_action
        )
        action_ds.attrs['timestamp'] = self.latest_action_timestamp

        # Increment the dataset index
        self.index += 1

        rospy.loginfo(f"Saved data index: {self.index}")

    def close(self):
        self.hdf5_file.close()

if __name__ == "__main__":
    rospy.init_node('data_collector', anonymous=True)
    dataset_dir = rospy.get_param("~dataset_dir", "./datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, "follow_cube.hdf5")
    collector = DataCollector(dataset_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        collector.close()
        rospy.loginfo("Data collection stopped.")
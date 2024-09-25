import torch
from models import ACTPolicy
from torchvision import transforms
import numpy as np
import argparse
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from collections import deque

class RobotController:
    def __init__(self, model, device, transform, sequence_length=5, num_joints=6):
        self.model = model
        self.device = device
        self.transform = transform
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.bridge = CvBridge()

        # Initialize ROS publishers
        self.target_pub = rospy.Publisher("/robot/next_target", Float64MultiArray, queue_size=10)

        # Subscribers for robot status and images
        rospy.Subscriber("/robot/status", Float64MultiArray, self.status_callback)
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

        # Queues to maintain sequences
        self.image_sequence = deque(maxlen=self.sequence_length)
        self.qpos_sequence = deque(maxlen=self.sequence_length)

        self.current_status = None
        self.lock = False  # To prevent concurrent access

    def status_callback(self, msg):
        self.current_status = np.array(msg.data)
        self.process_data()

    def image_callback(self, image_msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.current_image = cv_image
            self.process_data()
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def process_data(self):
        # Ensure both current_status and current_image are available
        if self.current_status is None or not hasattr(self, 'current_image'):
            return

        # Prevent concurrent access
        if self.lock:
            return
        self.lock = True

        try:
            # Apply transforms to the image
            img_transformed = self.transform(self.current_image)  # [3, H, W]
            img_transformed = img_transformed.to(self.device)

            # Prepare robot status
            qpos = torch.from_numpy(self.current_status).float().to(self.device)  # [num_joints + 3]

            # Append to sequences
            self.image_sequence.append(img_transformed)
            self.qpos_sequence.append(qpos)

            # Check if sequences have reached the required length
            if len(self.image_sequence) == self.sequence_length:
                # Stack sequences and add batch dimension
                imgs = torch.stack(list(self.image_sequence))  # [sequence_length, 3, H, W]
                qposes = torch.stack(list(self.qpos_sequence))  # [sequence_length, num_joints + 3]

                # Add batch dimension
                imgs = imgs.unsqueeze(1)  # [sequence_length, 1, 3, H, W]
                qposes = qposes.unsqueeze(1)  # [sequence_length, 1, num_joints + 3]

                # Move data to device
                imgs = imgs.to(self.device)
                qposes = qposes.to(self.device)

                # Predict next target position
                self.model.eval()
                with torch.no_grad():
                    predicted_actions = self.model(imgs, qposes)  # [sequence_length, 1, 3]

                # Get the last predicted action
                next_target = predicted_actions[-1, 0, :]  # [3]
                next_target = next_target.cpu().numpy()

                # Publish next target position
                target_msg = Float64MultiArray()
                target_msg.data = next_target.tolist()
                self.target_pub.publish(target_msg)
                rospy.loginfo(f"Published next target: {next_target}")

        except Exception as e:
            rospy.logerr(f"Error in process_data: {e}")
        finally:
            self.lock = False

def main(args):
    rospy.init_node('robot_inference_node', anonymous=True)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Initialize Model
    model = ACTPolicy(
        input_image_size=(224, 224),
        num_joints=args.num_joints,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_sequence_length=args.sequence_length
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load Checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    rospy.loginfo("Model loaded successfully.")

    # Initialize Robot Controller
    controller = RobotController(model, device, transform, sequence_length=args.sequence_length, num_joints=args.num_joints)

    rospy.loginfo("Inference node started. Waiting for data...")
    rospy.spin()

if __name__ == "__main__":
    import sys
    import cv2  # Ensure OpenCV is imported

    parser = argparse.ArgumentParser(description="Inference for ACT Policy")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--num_joints', type=int, default=6, help='Number of robot joints')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length for the ACT model')

    args = parser.parse_args()

    main(args)
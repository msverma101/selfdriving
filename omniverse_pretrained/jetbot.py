# Import necessary libraries and modules
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import cv2
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


class CoolController(BaseController):
    """
    A custom robot controller class that extends the BaseController.
    This controller uses an open-loop unicycle model for the robot's motion.

    Attributes:
        _wheel_radius (float): The radius of the robot's wheels.
        _wheel_base (float): The distance between the robot's wheels.
        step_ (int): A step counter for the controller.
    """

    def __init__(self):
        super().__init__(name="my_cool_controller")
        # An open loop controller that uses a unicycle model
        self._wheel_radius = 0.03
        self._wheel_base = 0.1125
        self.step_ = 0
        return

    def forward(self, joint_velocities):
        """
        Method to control the robot's motion.

        Args:
            joint_velocities (List[float]): List containing left and right wheel velocities.

        Returns:
            ArticulationAction: An object representing the robot's articulated action.
        """
            
        #left and right velocity
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=joint_velocities)



class Jetbot(BaseSample):
    """
    A sample class for Jetbot, a wheeled robot.

    Attributes:
        model (torch.nn.Module): The pre-trained PyTorch model for steering prediction.
        mean (torch.Tensor): Mean values used for image preprocessing.
        std (torch.Tensor): Standard deviation values used for image preprocessing.
    """

    def __init__(self, model_path, lego_asset_path) -> None:
        super().__init__()
        self.model = self.load_model(model_path)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        self.lego_asset_path = lego_asset_path
        return

    def setup_scene(self):
        """
        Method to set up the scene and add the Jetbot robot and Lego assets.
        """

        world = self.get_world()
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot =  WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position = np.array([0, 0.0, 0.03]),
                #orientation = np.array([0.70711, 0, 0, 0.70711])

                )
        world.scene.add(jetbot)
            
        add_reference_to_stage(self.lego_asset_path, prim_path = "/World/Lego")
        return

    async def setup_post_load(self):
        """
        Method to set up the Jetbot after the scene is loaded.
        """

        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_robot")
        self.camera_init()
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions) #(left_motor, right_motor)
        
        # Initialize our controller after load and the first reset
        self._my_controller = CoolController()

        return

    def send_robot_actions(self, step_size):
        """
        Callback function to send robot actions at each physics simulation step.

        Args:
            step_size (float): The time step of the physics simulation.
        """
                
        image = self.camera_observe()
        if image.shape[0]==0:
            return
        # self.display_image(image_path)
        left_motor, right_motor = self.execute(image)
        self._jetbot.apply_action(self._my_controller.forward(joint_velocities=[left_motor, right_motor])) #[left_motor, right_motor]
        position, orientation = self._jetbot.get_world_pose()
        print(orientation)

        return

    
    def camera_init(self):
        """
        Initialize the Jetbot's camera.
        """

        self.camera = Camera(prim_path="/World/Fancy_Robot/chassis/rgb_camera/jetbot_camera",resolution=(256, 256))
        self.camera.initialize()
        self.camera.add_motion_vectors_to_frame()
        return
    
    def camera_observe(self):
        """
        Observe the environment through the Jetbot's camera.

        Returns:
            np.ndarray: The camera image as a numpy array.
        """

        return self.camera.get_rgba()[:, :, :3]
    

    def load_model(self, path):

        """
        Load a pre-trained model for Jetbot's steering prediction.

        Args:
            path (str): The file path to the pre-trained model.

        Returns:
            torch.nn.Module: The loaded pre-trained model.
        """
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(path))
        self.device = torch.device("cuda")
        model = model.to(self.device)
        model = model.eval().half()
        return model


    def preprocess(self, image):
        """
        Preprocess the input image before feeding it to the steering prediction model.

        Args:
            image (np.ndarray): The camera image as a numpy array.

        Returns:
            torch.Tensor: The preprocessed image as a torch Tensor.
        """

        image = PIL.Image.fromarray(np.uint8(image))
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        return image[None, ...]
    

    def execute(self, image, angle = 0.0, angle_last = 0.0):

        """
        Execute the Jetbot's control algorithm.

        Args:
            image (np.ndarray): The camera image as a numpy array.
            angle (float, optional): Current steering angle. Default is 0.0.
            angle_last (float, optional): Previous steering angle. Default is 0.0.

        Returns:
            Tuple[float, float]: Left and right motor speeds calculated based on the control algorithm.
        """
        self.angle = angle
        self.angle_last = angle_last
        xy = self.model(self.preprocess(image)).detach().float().cpu().numpy().flatten()


        x = xy[0]
        y = (0.5 - xy[1]) / 2.0        
        # print(f'xy:{xy}', f'x:{x}', f'y:{y}' )
        
        # self.display_image(self.camera.get_rgba()[:, :, :3], x,y)
        self.save_image(xy)

        # Parameters for control
        speed_gain = 10.0 # Adjust this value for desired speed gain 0 to 1
        steering_gain = 10.0  # Adjust this value for desired steering gain 0 to 1
        steering_dgain = 0.0  # Adjust this value for desired steering kd 0 to 0.5
        steering_bias = 0.0  # Adjust this value for desired steering bias 0.3 to 0.3

        # Calculate angle and PID control
        angle = np.arctan2(x, y)
        pid = angle * steering_gain + (angle - angle_last) * steering_dgain
        angle_last = angle

        # Apply PID control to steering
        steering = pid + steering_bias

        # Calculate left and right motor speeds
        speed = speed_gain  # Adjust this value for desired speed
        left_motor = max(min(speed + steering, 10.0), 0.0)
        right_motor = max(min(speed - steering, 10.0), 0.0)
        # print(left_motor, right_motor)
        return left_motor, right_motor
        
    def save_image(self, xy):
        """
        Save the camera image with an annotated circle representing the predicted steering position.

        Args:
            xy (np.ndarray): Array containing x and y coordinates of the predicted steering position.
        """
        plt.imsave(rf"C:\Users\AI_Admin\Downloads\jetbot\Jebot\image\Images_from_omnivese\image_{xy[0]}_{xy[1]}.png",self.camera.get_rgba()[:, :, :3])
        # print("save_image")
        return
    

    def display_image(self, image, x, y):

        """
        Display the camera image with an annotated circle representing the predicted steering position.

        Args:
            image (np.ndarray): The camera image as a numpy array.
            x (float): x coordinate of the predicted steering position.
            y (float): y coordinate of the predicted steering position.
        """

        # Check if the image was successfully loaded
        if image is not None:
            # Create a window to display the image
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            # Display the image in the window
            cv2.imshow("Image", image)

            # Wait for a key press
            while True:
                key = cv2.waitKey(1) & 0xFF

                # Check if the 'q' key was pressed
                if key == ord('q'):
                    break

            # Close the window
            cv2.destroyAllWindows()
        else:
            print("Failed to load the image.")

from colosseum.rlbench.utils import ObservationConfigExt, name_to_class, check_and_make
from colosseum.rlbench.extensions.environment import EnvironmentExt
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from colosseum import (
    ASSETS_CONFIGS_FOLDER,
    ASSETS_JSON_FOLDER,
    TASKS_PY_FOLDER,
    TASKS_TTM_FOLDER,
)
import cv2
import os
import torch
from colosseum.tools.dataset_generator_onefile_fed import get_env_config
import matplotlib.pyplot as plt
from torchvision import transforms

def get_image_transform(config_file):
    transform = transforms.Compose([
        # transforms.Resize(64),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=config_file.transform.normalize_mean,
            std=config_file.transform.normalize_std
        )
    ])
    
    return transform

def process_obs(obs, transform=None):
    # invert axis from rgb to bgr
    front_image = torch.tensor(obs.front_rgb, dtype=torch.float32).permute(2, 0, 1)/255
    front_image = transform(front_image)

    # process observations for agent
    low_dim_state = torch.tensor(np.concatenate((obs.joint_positions, np.array([obs.gripper_open]))), dtype=torch.float32)

    return front_image, low_dim_state

def reverse_process_image(image):
    image = (image * 0.5) + 0.5
    image = (image.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    return image

def normalize_action(action, action_min, action_max):
    # Scale action to [0, 1] then map to [-1, 1]
    normalized_action = 2 * ((action - action_min) / (action_max - action_min)) - 1
    return normalized_action

def denormalize_action(normalized_action, action_min, action_max):
    # Map from [-1, 1] back to [0, 1] then to the original range
    action = ((normalized_action + 1) / 2) * (action_max - action_min) + action_min
    return action

def save_video_trajectory(front_images, video_path='/home/omniverse/Workspace/elsa_robotic_manipulation/elelsa_robotic_manipulationsa/videos', video_name='slide_block_to_target.mp4'):
        # create a video

    os.makedirs(video_path, exist_ok=True)

    video_path = os.path.join(video_path, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (128, 128))

    for img in front_images:
        img = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()

    # video to gif
    import moviepy.editor as mp
    clip = mp.VideoFileClip(video_path)
    clip.write_gif(video_path.replace('mp4', 'gif'))    

    
    print(f"Video saved at {video_path}")


def load_environment(base_cfg, collection_cfg, idx_environment, headless=True):
    """ Load the environment with the given configuration.

    Args:
        base_cfg (BaseConfig): The base configuration.
        collection_cfg (CollectionConfig): The collection configuration.
        idx_environment (int): The index of the environment to load.
        headless (bool): Whether to run the environment in headless mode.

    Returns:
        EnvironmentExt: The loaded environment.
    """
    
    task = name_to_class(base_cfg.env.task_name, TASKS_PY_FOLDER)
    config = get_env_config(base_cfg, collection_cfg, idx_environment)

    data_cfg, env_cfg = config.data, config.env

    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_cfg),
        headless=headless,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=env_cfg,
        )

    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)

    return task_env, rlbench_env

def load_config():
    config_path = os.path.join("./dataset_config.yaml")
    config = OmegaConf.load(config_path)
    return config
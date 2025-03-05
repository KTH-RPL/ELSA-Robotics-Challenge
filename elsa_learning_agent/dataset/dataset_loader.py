import os
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from elsa_learning_agent.utils import process_obs, normalize_action, get_image_transform
from colosseum.rlbench.datacontainer import DataContainer


class ImitationDataset(Dataset):
    def __init__(self, config, train=False, test=False, normalize=False):
        self.root_dir = config.dataset.root_dir
        task = config.dataset.task
        env_id = config.dataset.env_id
        train_split = config.dataset.train_split
        data_path = os.path.join(self.root_dir, f"{task}", f"env_{env_id}", "episodes_observations.pkl.gz")
        
        datacontainer = DataContainer()
        demos_raw_data = datacontainer.load(data_path).data
        self.normalize = normalize
        self.action_min = torch.tensor(config.transform.action_min)
        self.action_max = torch.tensor(config.transform.action_max)

        if train:
            demos_raw_data = demos_raw_data[:int(train_split * len(demos_raw_data))]
        elif test:
            demos_raw_data = demos_raw_data[int(config.dataset.test_split * len(demos_raw_data)):]

        self.transform = get_image_transform(config)
        self.data = []
        self.demos_idx = []

        # Load data
        print("Loading dataset from:", data_path)
        for i, demo in enumerate(tqdm(demos_raw_data)):
            self.demos_idx.append(len(self.data))
            num_steps = len(demo) - 1
            for t in range(num_steps):
                self.data.append(self._load_datapoint(demo, t))

    def _load_datapoint(self, trajectory, time_step):
        obs = trajectory[time_step]
        next_obs = trajectory[time_step + 1]
        front_image, low_dim_state = process_obs(obs, self.transform)
        action = torch.tensor(
            np.concatenate((obs.joint_velocities, np.array([next_obs.gripper_open]))), 
            dtype=torch.float32
        )
        if self.normalize:
            action = normalize_action(action, self.action_min, self.action_max)
        return {
            "action": action,
            "low_dim_state": low_dim_state,
            "image": front_image,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    # Manually set the config directory outside the script folder
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))

    # Ensure Hydra finds the external config directory
    @hydra.main(config_path=config_path, config_name="config", version_base=None)
    def run(config: DictConfig):
        print("Loaded Configuration:")
        print(OmegaConf.to_yaml(config))  # Print config for debugging
        
        #TO DEBUG:
        # config.dataset.root_dir = "/home/omniverse/Workspace/elsa_robotic_manipulation/eval_data"
        # config.dataset.env_id = 0
        dataset = ImitationDataset(config, test=False, normalize=True)
        print("Dataset size:", len(dataset))
        sample = dataset[0]

        # Plot sample image
        print("Low-Dim Data:", sample["low_dim_state"])
        print("Action:", sample["action"])
        print("Image shape:", sample["image"].shape)

        # plot the image
        # inverse normalization
        sample["image"] = (sample["image"] * 0.5) + 0.5
        plt.imshow(sample["image"].permute(1, 2, 0).numpy())
        plt.show()

        # compute mean and std of the dataset for the images for normalization
        mean = 0.
        std = 0.
        nb_samples = 0.

        for sample in dataset:
            image = sample["image"]
            batch_samples = image.size(0)
            image = image.view(3, -1)
            mean += image.mean(1)
            std += image.std(1)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        print("Mean:", mean)
        print("Std:", std)

        # compute mean and std of actions
        actions = []
        for sample in dataset:
            actions.append(sample["action"].numpy())
        actions = np.array(actions)
        mean = np.mean(actions, axis=0)
        std = np.std(actions, axis=0)
        print("Mean:", mean)
        print("Std:", std)

    run()  # Call the Hydra-wrapped function


if __name__ == "__main__":
    main()

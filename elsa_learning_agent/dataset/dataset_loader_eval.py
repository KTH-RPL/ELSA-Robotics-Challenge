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


class EvalImitationDataset(Dataset):
    def __init__(self, config, train=False, test=False):
        self.root_dir = config.dataset.root_eval_dir
        task = config.dataset.task
        env_id = config.dataset.env_id
        data_path = os.path.join(self.root_dir, f"{task}", f"env_{env_id}", "episodes_observations.pkl.gz")
        
        datacontainer = DataContainer()
        obs_raw_data = datacontainer.load(data_path)["data"]

        self.transform = get_image_transform(config)
        self.data = []
        self.demos_idx = []

        # Load data
        print("Loading dataset from:", data_path)
        for key, obs in tqdm(obs_raw_data.items()):
            self.data.append(self._load_datapoint(obs))

    def _load_datapoint(self, obs):
        front_image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1)/255
        front_image = self.transform(front_image)
        low_dim_state = torch.tensor(obs["low_dim_state"], dtype=torch.float32)
        obs_id = obs["obs_id"]

        return {
            "obs_id": obs_id,
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

        dataset = EvalImitationDataset(config, test=True)
        print("Dataset size:", len(dataset))
        sample = dataset[0]

        # Plot sample image
        print("Low-Dim Data:", sample["low_dim_state"])
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

    run()  # Call the Hydra-wrapped function


if __name__ == "__main__":
    main()

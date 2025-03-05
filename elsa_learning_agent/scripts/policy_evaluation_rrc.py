import signal
import sys
import torch
import os
import numpy as np
import pickle
from omegaconf import DictConfig, OmegaConf
import hydra
from colosseum import (
    ASSETS_CONFIGS_FOLDER,
)
from elsa_learning_agent.agent import Agent
from elsa_learning_agent.dataset.dataset_loader_eval import EvalImitationDataset
from elsa_learning_agent.utils import denormalize_action
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import argparse

def validate_one_epoch(agent, val_loader, cfg, task):
    agent.eval()
    predicted_actions= {f"{task}": []}
    image_counter = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch["image"].to("cuda")
            low_dim_state = batch["low_dim_state"].to("cuda")
            obs_ids = batch["obs_id"]

            predicted_action = agent.get_action(image, low_dim_state)

            for idx, action in enumerate(predicted_action):
                denormalized_action = denormalize_action(action.detach().cpu(), torch.tensor(cfg.transform.action_min), torch.tensor(cfg.transform.action_max))
                datapoint = {"action_id": int(obs_ids[idx]),  "action": denormalized_action.numpy().tolist()}
                predicted_actions[f"{task}"].append(datapoint)
                image_counter += 1


    return predicted_actions


def main(model_paths, model_config_path, predictions_path) -> int:

    # load json containin all default predictions
    all_predictions = {"data": {}}

    for task in ["slide_block_to_target", "close_box", "scoop_with_spatula", "insert_onto_square_peg"]:
        model_path = model_paths[task]
        if model_path is None:
            print("No model path provided for task: ", task) 
            print("Please provide the weights for task: ", task) 
            print("Closing the script...")     
            os.kill(os.getppid(),signal.SIGTERM)
            sys.exit(2)
        else:

            model_cfg = OmegaConf.load(model_config_path)
            model_cfg.dataset.dataset_task = task
            model_cfg.dataset.task = task

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_cfg.dataset.root_eval_dir = model_cfg.dataset.root_test_dir

            dataset = EvalImitationDataset(model_cfg, train=False, test=True)
            # Important not to shuffle the data
            dataloader = DataLoader(dataset, batch_size=model_cfg.dataset.batch_size, shuffle=True)

            agent = Agent(
                image_channels=3,
                low_dim_state_dim=8,
                action_dim=8,
                image_size=(128,128)
            )
            agent.policy.to(device)

            print(f"Loading model from: {model_path}")
            agent.load_state_dict(model_path)

            print("Starting evaluation...")
            predctions = validate_one_epoch(agent, dataloader, model_cfg, task)

            all_predictions["data"][task] = predctions[task]

    # find the dir of the default predictions
    save_action_path =  os.path.join(os.path.dirname(predictions_path),  "actions.json")

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_action_path), exist_ok=True)

    with open(save_action_path, "w") as f:
        json.dump(all_predictions, f, indent=4)
    
    print(f"Saved actions dataset to: {save_action_path}")


if __name__ == "__main__":
    # TODO: set the correct model paths with your best trained model for each task
    model_paths = {
        "slide_block_to_target": "./model_checkpoints/slide_block_to_target/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth", 
        "close_box": "./model_checkpoints/close_box/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth", 
        "scoop_with_spatula": "./model_checkpoints/scoop_with_spatula/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth",  
        "insert_onto_square_peg": "./model_checkpoints/insert_onto_square_peg/Avg_BCPolicy_l-ep_50_ts_0.9_fclients_0.05_round_30.pth", 
        }
    
    
    model_config_path = "./dataset_config.yaml"
    
    # Set the path to save the predictions
    predictions_path = "./results/predictions.json"
    main(model_paths, model_config_path, predictions_path)
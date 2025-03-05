import ray
import torch
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from federated_elsa_robotics.task import validate_one_epoch
from elsa_learning_agent.agent import Agent
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.utils import get_image_transform
from elsa_learning_agent.utils import load_environment, process_obs, denormalize_action, get_image_transform

import pickle
import argparse
import time
import numpy as np
import json

# Define how many parallel jobs to run per GPU
NUM_GPUS = torch.cuda.device_count()
JOBS_PER_GPU = 10  # Adjust based on GPU memory availability


def online_evaluation(agent, device, transform, base_cfg, idx_environment, num_episodes=5):
    agent.eval()

        # Load the configuration
    collection_cfg_path: str = (
        os.path.join(base_cfg.dataset.root_dir, base_cfg.env.task_name, base_cfg.env.task_name) + "_fed.json"
    )
    with open(collection_cfg_path, "r") as fh:
        collection_cfg = json.load(fh)

    task_env, rlbench_env = load_environment(base_cfg, collection_cfg, idx_environment, headless=True)

    rewards = []
    best_reward = -float("inf")

    # save all the images to then create a video
    best_images = []
    for i in range(num_episodes):
        front_images = []
        total_reward = 0.0
        descriptions, obs = task_env.reset()
        front_images.append(obs.front_rgb)
        terminate = False
        t = 0
        while not terminate and t < 300:
            # prcess observations for agent
            front_rgb, low_dim_state = process_obs(obs, transform)
            front_rgb = front_rgb.unsqueeze(0).to(device)
            low_dim_state = low_dim_state.unsqueeze(0).to(device)

            action = agent.get_action(front_rgb, low_dim_state)
            denormalized_action = denormalize_action(action.detach().cpu(), torch.tensor(base_cfg.transform.action_min), torch.tensor(base_cfg.transform.action_max))
            obs, reward, terminate = task_env.step(denormalized_action.numpy()[0])
            front_images.append(obs.front_rgb)
            t += 1
            total_reward += reward

        # for now only keep track of the final reward
        rewards.append(reward)

        if reward > best_reward:
            best_reward = reward
            best_images = front_images


    # Convert to numpy array of shape (T, H, W, C)
    video_array = np.stack(best_images, axis=0)

    # Convert to format required by wandb.Video (T, C, H, W)
    video_array = video_array.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    rlbench_env.shutdown()

    return rewards



def save_table(model_rounds, rmse_values, rmse_std_values, success_rates, success_rates_std, txt_file_path):
    # Generate a formatted table manually without using PrettyTable

    # Define column headers
    table_str = f"{'Model Round':<15}{'RMSE':<15}{'RMSE std':<15}{'Success Rate':<15}{'Success Std':<15}\n"
    table_str += "=" * 60 + "\n"  # Adjusted separator line for new column

    # Add rows
    for round_num, rmse, rmse_std, success, success_std in zip(model_rounds, rmse_values, rmse_std_values, success_rates, success_rates_std):
        table_str += f"{round_num:<15}{rmse:<15.4f}{rmse_std:<15.4f}{success:<15.4f}{success_std:<15.4f}\n"

    # Print the table
    print(table_str)

    with open(txt_file_path, "w") as f:
        f.write(table_str)

    # Output confirmation of saved file
    print(f"Saved results table to {txt_file_path}")


@ray.remote
def evaluate_single_env(idx_evaluate, net_args, model_path, dataset_config, base_cfg):
    """Function to run evaluation for a single environment"""
    print(f"Running online evaluation for env {idx_evaluate}")
    # Load model
    agent = Agent(**net_args)
    agent.load_state_dict(model_path)
    rewards = online_evaluation(
        agent, "cpu", get_image_transform(dataset_config), base_cfg, idx_evaluate, dataset_config.dataset.num_episodes_live
    )
    return rewards


def test_model(current_round, model_path, test_dataset, device, net_args, simulator, dataset_config):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # Ensure correct GPU allocation
    print(f"Running evaluation on device {device}")

    """Evaluate a model on a test dataset, running multiple jobs per GPU."""
    # Load model
    agent = Agent(**net_args)
    agent.load_state_dict(model_path)
    agent.policy.to(f"cuda:{device}")


    metrics = {}
    metrics["loss_per_env"] = {}

    loss_values = []  # Store individual loss values

    for idx, dataset in enumerate(test_dataset):
        print(f"Running offline evaluation for env {idx}")
        
        # Compute RMSE (loss)
        loss = validate_one_epoch(agent, dataset, device=f"cuda:{device}")
        
        # Store the loss value
        loss_values.append(loss)
        
        # Track loss per environment
        metrics["loss_per_env"][dataset_config.dataset.final_test_env_idx_range[0] + idx] = loss

    # Compute mean and standard deviation
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)  # Use ddof=1 for sample std deviation


    if simulator:
        print("Loading config from: ", dataset_config.dataset.root_dir + f"/{dataset_config.dataset.task}/{dataset_config.dataset.task}_fed.yaml")
        base_cfg = OmegaConf.load(
            dataset_config.dataset.root_dir + f"/{dataset_config.dataset.task}/{dataset_config.dataset.task}_fed.yaml"
        )
        base_cfg.dataset = dataset_config.dataset
        base_cfg.transform = dataset_config.transform


        ################################ PARALLELISM WITH RAY ################################
        # Submit parallel jobs for each environment using Ray
        futures = [
            evaluate_single_env.remote(idx, net_args, model_path, dataset_config, base_cfg)
            for idx in dataset_config.dataset.final_test_live_idxs
        ]

        results = ray.get(futures)  # Retrieve results

        # Aggregate results, now there are lists of lists as we have all the rewards for each environments, and we want to compute the average and std
        flattened_rewards = np.concatenate(results)

        # Compute mean and standard deviation
        mean_reward = np.mean(flattened_rewards)
        std_reward = np.std(flattened_rewards)
        
        metrics = {"mean_reward": mean_reward, "std_reward": std_reward}


    return current_round, mean_loss, std_loss, metrics



@ray.remote(num_gpus=1 / JOBS_PER_GPU)  # Allow multiple j"obs per GPU
def evaluate_model(current_round, model_path, test_dataset, device, net_args, simulator, dataset_config):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # Ensure correct GPU allocation
    print(f"Running evaluation on device {device}")

    """Evaluate a model on a test dataset, running multiple jobs per GPU."""
    # Load model
    agent = Agent(**net_args)
    agent.load_state_dict(model_path)
    agent.policy.to(f"cuda:{device}")


    metrics = {}
    metrics["loss_per_env"] = {}

    loss_values = []  # Store individual loss values

    for idx, dataset in enumerate(test_dataset):
        print(f"Running offline evaluation for env {idx}")
        
        # Compute RMSE (loss)
        loss = validate_one_epoch(agent, dataset, device=f"cuda:{device}")
        
        # Store the loss value
        loss_values.append(loss)
        
        # Track loss per environment
        metrics["loss_per_env"][dataset_config.dataset.final_test_env_idx_range[0] + idx] = loss

    # Compute mean and standard deviation
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)

    if simulator:
        print("Loading config from: ", dataset_config.dataset.root_dir + f"/{dataset_config.dataset.task}/{dataset_config.dataset.task}_fed.yaml")
        base_cfg = OmegaConf.load(
            dataset_config.dataset.root_dir + f"/{dataset_config.dataset.task}/{dataset_config.dataset.task}_fed.yaml"
        )
        base_cfg.dataset = dataset_config.dataset
        base_cfg.transform = dataset_config.transform

        ################################ PARALLELISM WITH RAY ################################
        # Submit parallel jobs for each environment using Ray
        futures = [
            evaluate_single_env.remote(idx, net_args, model_path, dataset_config, base_cfg)
            for idx in dataset_config.dataset.final_eval_live_idxs
        ]

        results = ray.get(futures)  # Retrieve results

        # Aggregate results, now there are lists of lists as we have all the rewards for each environments, and we want to compute the average and std
        flattened_rewards = np.concatenate(results)

        # Compute mean and standard deviation
        mean_reward = np.mean(flattened_rewards)
        std_reward = np.std(flattened_rewards)

        metrics = {"mean_reward": mean_reward, "std_reward": std_reward}


    return current_round, mean_loss, std_loss, metrics



def main(task="slide_block_to_target", local_epochs=100, rounds_to_evaluate=20, fraction_fit=0.05, train_test_split=0.9):
    print(f"Running evaluation for task {task} with {rounds_to_evaluate} rounds. Params {local_epochs} local epochs, train_split {train_test_split}, and {fraction_fit} fraction of clients.")
    start_time = time.time()
    # get omegaconf config
    ###############
    dataset_config_path = f"elsa_robotics/dataset_config_{task}.yaml"
    print(f"Loading dataset config from {dataset_config_path}")
    config = OmegaConf.load(dataset_config_path)

    net_args = {
        "image_channels": 3,
        "low_dim_state_dim": 8,
        "action_dim": 8,
        "image_size": (128, 128),
    }

    model_name = f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}_round_"  # Adjust as needed
    model_paths = [os.path.join("model_checkpoints", config.dataset.task, model_name + f"{i+1}.pth") for i in range(rounds_to_evaluate)]

    def create_config(idx): 
        cur_config = OmegaConf.load(dataset_config_path)
        cur_config.dataset.env_id = idx
        # Use evaluation dataset for the server
        cur_config.dataset.root_dir = cur_config.dataset.root_eval_dir
        cur_config.dataset.train_split = 0.1        # load only 10 percent of the data as we evaluate on 10 new environments
        cur_config.dataset.test_split = 0.0
        cur_config.dataset.num_server_rounds = rounds_to_evaluate
        cur_config.dataset.local_epochs = local_epochs
        return cur_config

    config = create_config(0)
    eval_dataset = [
            DataLoader(ImitationDataset(config=create_config(idx), train=True), batch_size=32, shuffle=False, num_workers=64)
        for idx in range(*config.dataset.final_eval_env_idx_range)]


    # Get assigned GPU IDs from CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        assigned_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    else:
        assigned_gpus = list(range(torch.cuda.device_count()))  # Fallback

    print(f"Assigned GPUs: {assigned_gpus}")  # Debugging

    device_list = assigned_gpus
    # print(f"device list: {device_list}")


    # Submit initial batch of jobs
    futures = [
        evaluate_model.remote(i, model_path, eval_dataset, 0, net_args, True, config)
        for i, model_path in enumerate(model_paths[:JOBS_PER_GPU])
    ]

    # Process results in batches
    results = []
    remaining_models = model_paths[JOBS_PER_GPU:]

    while futures:
        # Wait for any job to complete
        done, futures = ray.wait(futures, num_returns=1)

        # Collect the result from the completed job
        results.extend(ray.get(done))

        # If there are remaining models, submit the next one
        if remaining_models:
            next_model_path = remaining_models.pop(0)
            next_future = evaluate_model.remote(
                len(results) + len(futures), next_model_path, eval_dataset, 0, net_args, True, config
            )
            futures.append(next_future)


    # Process results and sort by model_round
    sorted_results = sorted(results, key=lambda x: x[0])  # Sort by model_round

    # Extract sorted data
    model_rounds = [res[0] for res in sorted_results]
    print(model_rounds)
    rmse_values = [res[1] for res in sorted_results]
    rmse_std = [res[1] for res in sorted_results]
    success_rates = [res[3].get("mean_reward", 0) for res in sorted_results]
    success_rates_std = [res[3].get("std_reward", 0) for res in sorted_results]

    # get the directory of the model checkpoints
    model_dir = os.path.dirname(model_paths[0])

    # Generate new filenames for saving
    plot_rmse_path = os.path.join(model_dir.replace("model_checkpoints", "results"),  f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}", "rmse.png")
    plot_success_path = os.path.join(model_dir.replace("model_checkpoints", "results"), f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}", "success_rate.png")
    results_pickle_path = os.path.join(model_dir.replace("model_checkpoints", "results"), f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}", "results.pkl")
    txt_file_path = os.path.join(model_dir.replace("model_checkpoints", "results"), f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}", "results.txt")
    time_file_path = os.path.join(model_dir.replace("model_checkpoints", "results"), f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}", "time.txt")

    # Prepare directory for results
    os.makedirs(os.path.dirname(plot_rmse_path), exist_ok=True)

    # Create RMSE plot
    plt.figure(figsize=(10, 5))
    plt.plot(model_rounds, rmse_values, marker="o", linestyle="-")
    plt.xlabel("Model Round")
    plt.ylabel("RMSE")
    plt.title("RMSE per Model Round")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(plot_rmse_path)
    plt.close()

    # Create Success Rate plot
    plt.figure(figsize=(10, 5))
    plt.plot(model_rounds, success_rates, marker="o", linestyle="-", color="green")
    plt.xlabel("Model Round")
    plt.ylabel("Success Rate")
    plt.title("Success Rate per Model Round")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(plot_success_path)
    plt.close()

    save_table(model_rounds, rmse_values,rmse_std, success_rates, success_rates_std, txt_file_path)

    with open(results_pickle_path, "wb") as f:
        pickle.dump(results, f)

    # Output confirmation of saved files
    print(f"Saved RMSE plot to {plot_rmse_path}")
    print(f"Saved Success Rate plot to {plot_success_path}")
    print(f"Saved aggregated results to {results_pickle_path}")


    ####################### FINAL TEST EVALUATION on BEST MODEL #######################
    best_round = success_rates.index(max(success_rates))
    best_model_path = model_paths[best_round]

    test_dataset = [
            DataLoader(ImitationDataset(config=create_config(idx), train=True), batch_size=32, shuffle=False, num_workers=64)
        for idx in range(*config.dataset.final_test_env_idx_range)]

    current_round, mean_loss, std_loss, metrics = test_model(best_round, best_model_path, test_dataset, 0, net_args, True, config)

    # create a text file to save these final results
    final_results_txt_path = os.path.join(model_dir.replace("model_checkpoints", "results"), f"BCPolicy_l-ep_{local_epochs}_ts_{train_test_split}_fclients_{fraction_fit}", "final_test_results.txt")

    print(f"Best round: {best_round}")
    print(f"Best model path: {best_model_path}")
    print(f"Mean loss: {mean_loss}")
    print(f"Std loss: {std_loss}")
    print(f"Mean success rate: {metrics['mean_reward']}")
    print(f"Std success rate: {metrics['std_reward']}")
    with open(final_results_txt_path, "w") as f:
        f.write(f"Best round: {best_round}\n")
        f.write(f"Best model path: {best_model_path}\n")
        f.write(f"Mean loss: {mean_loss}\n")
        f.write(f"Std loss: {std_loss}\n")
        f.write(f"Mean success rate: {metrics["mean_reward"]}\n")
        f.write(f"Std success rate: {metrics["std_reward"]}\n")
    print(f"Final test results saved to {final_results_txt_path}")

    final_time = (time.time() - start_time)/60

    print(f"Total time taken: {final_time:.2f} minutes")
    # save this to a file
    with open(time_file_path, "a") as f:
        f.write(f"Total evaluation time taken: {final_time:.2f} minutes\n\n")
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Behavior Cloning Policy with given parameters.")

    parser.add_argument("--task", type=str, default="slide_block_to_target", help="Type of input modality (e.g., 'slide_block_to_target', close_box)")
    parser.add_argument("--local_epochs", type=int, default=100,  help="Number of local training epochs")
    parser.add_argument("--rounds_to_evaluate", type=int, default=10,  help="Number of rounds to evaluate")
    parser.add_argument("--fraction_fit", type=float, default=0.05, help="Fraction of clients participating in federated training")
    parser.add_argument("--train_test_split", type=float, default=1.0, help="Train test split")
    parser.add_argument("--plotting", action="store_true", help="Plotting the results")

    args = parser.parse_args()

    main(
        modality=args.modality,
        task=args.task,
        local_epochs=args.local_epochs,
        rounds_to_evaluate=args.rounds_to_evaluate,
        fraction_fit=args.fraction_fit,
        train_test_split=args.train_test_split
    )

"""elsa-robotics: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from federated_elsa_robotics.strategies import SaveModelStrategy
from federated_elsa_robotics.task import get_weights, set_weights, validate_one_epoch
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from elsa_learning_agent.agent import Agent
from federated_elsa_robotics.eval_model import online_evaluation
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.utils import get_image_transform

def gen_evaluate_fn(
    testloader: list[ImitationDataset],
    device: torch.device,
    net_args: dict,
    simulator: bool = False,
    dataset_config: dict = None,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        agent = Agent(**net_args)
        set_weights(agent, parameters_ndarrays)
        agent.policy.to(device)

        metrics = {}

        total_loss = 0.0
        metrics["loss_per_env"] = {}
        for idx, dataset in enumerate(testloader):
            loss = validate_one_epoch(agent, dataset, device=device)
            total_loss += loss
            metrics["loss_per_env"][dataset_config.dataset["test_env_idx_range"][0] + idx] = loss
        
        loss = total_loss / len(testloader)

        if simulator:
            # Load base config from dataset yaml
            base_cfg = OmegaConf.load(dataset_config.dataset["root_dir"] + f"/{dataset_config.dataset['task']}/{dataset_config.dataset['task']}_fed.yaml")
            base_cfg.dataset = dataset_config.dataset
            base_cfg.transform = dataset_config.transform

            metrics["avg_reward"] = 0.0
            metrics["video_array"] = []
            for idx_evaluate in dataset_config.dataset["test_live_idxs"]:                
                # Perform online evaluation
                avg_reward, video_array = online_evaluation(agent, device, get_image_transform(dataset_config), base_cfg, idx_evaluate, dataset_config.dataset["num_episodes_live"])
                metrics["avg_reward"] += avg_reward
                metrics["video_array"].append(video_array)
            metrics["avg_reward"] /= len(dataset_config.dataset["test_live_idxs"])
            
        return loss, metrics

    return evaluate


def train_aggregation_fn(metrics: list[dict]):
    """Aggregate training metrics."""
    losses = [num_samples * m["train_loss"] for num_samples, m in metrics]
    examples = [num_samples for num_samples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}

def server_fn(context: Context):
    # Read from project toml config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-eval"]
    server_device = context.run_config["server-device"]
    use_wandb = context.run_config["use-wandb"]
    dataset_config_path = context.run_config["dataset-config-path"]
    conf = OmegaConf.load(dataset_config_path)
    print(f"Starting server with l-ep={context.run_config["local-epochs"]}, ts={context.run_config["train-split"]}, fclients={fraction_fit}")

    net_args = {
        "image_channels": 3,
        "low_dim_state_dim": 8,
        "action_dim": 8,
        "image_size": (128, 128),
    }

    # Initialize model parameters
    agent = Agent(**net_args)
    ndarrays = get_weights(agent)
    parameters = ndarrays_to_parameters(ndarrays)

    # Evaluation loader
    def create_config(idx): 
        cur_config = conf.copy()
        cur_config.dataset.task = context.run_config["dataset-task"]
        cur_config.dataset.env_id = idx
        # Use evaluation dataset for the server
        cur_config.dataset.root_dir = cur_config.dataset.root_eval_dir
        cur_config.dataset.test_split = 0.0
        cur_config.dataset.train_split = context.run_config["train-split"]
        cur_config.dataset.num_server_rounds = num_rounds
        cur_config.dataset.local_epochs = context.run_config["local-epochs"]
        return cur_config
    config = create_config(0)
    # test_dataset = [
    #         DataLoader(ImitationDataset(config=create_config(idx), test=True), batch_size=config.dataset["batch_size"], shuffle=False, num_workers=config.dataset["num_workers_server"])
    #     for idx in range(*config.dataset["test_env_idx_range"])]
    
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=parameters,
        agent=agent,
        config=config,
        use_wandb=use_wandb,
        # evaluate_fn=gen_evaluate_fn(test_dataset, server_device, net_args, simulator=config.dataset["enable_live_eval"], dataset_config=config),
        # evaluate_metrics_aggregation_fn=
        fit_aggregation_fn=train_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def get_server_app():
    return ServerApp(server_fn=server_fn)
# Create ServerApp
app = ServerApp(server_fn=server_fn)

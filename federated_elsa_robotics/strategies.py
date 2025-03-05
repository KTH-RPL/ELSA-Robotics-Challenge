from collections import OrderedDict
from typing import Optional, Union
from flwr.common import FitRes, parameters_to_ndarrays, Context, Parameters, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server import client_proxy
from flwr.server.strategy import FedAvg
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
import wandb
from elsa_learning_agent.agent import Agent


class SaveModelStrategy(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_clients: int = 2,
        initial_parameters = None,
        agent: Agent = Agent(),
        save_path: Path = Path("model_checkpoints"),
        config: Optional[OmegaConf] = None,
        use_wandb: bool = False,
        evaluate_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        fit_aggregation_fn: Optional[callable] = None,
        resume: bool = False,
    ):
            
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_aggregation_fn,
        )
        self.agent = agent
        self.save_name = f"{agent.policy.__class__.__name__}_l-ep_{config.dataset["local_epochs"]}_ts_{config.dataset["train_split"]}_fclients_{fraction_fit}"
        # Chkpt/task/wandbName
        self.save_path = Path.joinpath(save_path, config.dataset["task"])
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        if self.use_wandb:
            if resume:
                wandb.init(
                    project="BCPolicy-Training",  # Replace with your project name
                    # name=f"{base_cfg.dataset.task}_bc_env{idx_environment}_{train_split}",       # Replace with a run name
                    name=f"{config.dataset['task']}_{self.save_name}",
                    id="run-20250226_101133-uoligt1l",
                    resume="must",
                    config={
                        "client_epochs": config.dataset["local_epochs"],
                        "train_split": config.dataset["train_split"],
                        "fitted_clients": fraction_fit,
                        "batch_size": config.dataset["batch_size"],
                        "rounds": config.dataset["num_server_rounds"],
                        "learning_rate": config.model["learning_rate"],
                        "weight_decay": config.model["weight_decay"],
                    }
                )
            else:
                wandb.init(
                    project="BCPolicy-Training",  # Replace with your project name
                    # name=f"{base_cfg.dataset.task}_bc_env{idx_environment}_{train_split}",       # Replace with a run name
                    name=f"{config.dataset['task']}_{self.save_name}",
                    config={
                        "client_epochs": config.dataset["local_epochs"],
                        "train_split": config.dataset["train_split"],
                        "fitted_clients": fraction_fit,
                        "batch_size": config.dataset["batch_size"],
                        "rounds": config.dataset["num_server_rounds"],
                        "learning_rate": config.model["learning_rate"],
                        "weight_decay": config.model["weight_decay"],
                    }
                )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[client_proxy.ClientProxy, FitRes]],
        failures: list[Union[tuple[client_proxy.ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(self.agent.policy.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.agent.policy.load_state_dict(state_dict)

            # Save the model to disk
            path = Path.joinpath(self.save_path, f"{self.save_name}_round_{server_round}.pth")
            torch.save(self.agent.policy.state_dict(), path)

            # Save the parameters to wandb
            if self.use_wandb:
                wandb.log({
                    "server_round": server_round,
                    "federated_training_loss": aggregated_metrics["train_loss"]})

        return aggregated_parameters, aggregated_metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """ Centralized evaluation callback"""

        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        centr_loss, centr_metric = super().evaluate(server_round, parameters)

        # Log the centralized evaluation loss to wandb
        if self.use_wandb:
            centralized_eval_metric = {
                "loss_per_env": centr_metric["loss_per_env"],
            }
            if centr_metric.get("avg_reward") is not None:
                centralized_eval_metric["avg_reward"] = centr_metric["avg_reward"]
                centralized_eval_metric["video_array"] = [
                    wandb.Video(video, fps=20, format="gif") for video in centr_metric["video_array"]
                ]

            wandb.log({
                    "server_round": server_round,
                    "centralized_eval_loss": centr_loss,
                    "centralized_eval_metric": centralized_eval_metric
                    })

        return centr_loss, centr_metric
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation metrics from all clients"""
        fed_eval_loss, fed_eval_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Log the federated evaluation loss to wandb
        if self.use_wandb:
            wandb.log({
                        "server_round": server_round,
                        "federated_eval_loss": fed_eval_loss
                    })

        return fed_eval_loss, fed_eval_metrics


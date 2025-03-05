"""elsa-robotics: A Flower / PyTorch app."""

from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from elsa_learning_agent.dataset.dataset_loader import ImitationDataset
from elsa_learning_agent.agent import Agent


def load_data_colosseum(partition_id: int, num_partitions: int, train_split : float = 0.9, config: dict = None):
    """Load partition Colosseum data."""
    # Load the dataset
    train_dataset = ImitationDataset(config=config, train=True)
    print(f"For partition_id={partition_id}, len(train_dataset): {len(train_dataset)}")

    val_dataset = ImitationDataset(config=config, test=True)
    print(f"For partition_id={partition_id}, len(val_dataset): {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False)
    return train_loader, val_loader

# Training and validation loop
def train_one_epoch(agent: Agent, train_loader, optimizer, criterion, epoch, device):
    agent.train()
    total_loss = 0.0
    for batch in train_loader:
        image = batch["image"].to(device)
        low_dim_state = batch["low_dim_state"].to(device)
        action = batch["action"].to(device)

        optimizer.zero_grad()
        predicted_action = agent.get_action(image, low_dim_state)
        loss = criterion(predicted_action, action)
        loss.backward()
        optimizer.step()

        total_loss += math.sqrt(loss.item())  # RMSE loss

    avg_loss = total_loss / len(train_loader)
    # print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(agent:Agent, val_loader, device):
    criterion = torch.nn.MSELoss()
    agent.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            low_dim_state = batch["low_dim_state"].to(device)
            action = batch["action"].to(device)

            predicted_action = agent.get_action(image, low_dim_state)
            loss = criterion(predicted_action, action)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    # print(f"Epoch {epoch}: Validation Loss = {avg_loss:.4f}")
    return avg_loss

def train(agent: Agent, trainloader, epochs, device, config):
    """Train the model on the training set."""
    agent.policy.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()  # Behavioral Cloning uses MSE loss
    optimizer = optim.Adam(agent.policy.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    # Training loop
    running_loss = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(agent, trainloader, optimizer, criterion, epoch, device)
        running_loss += train_loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    avg_trainloss = running_loss / epochs
    return avg_trainloss

def get_weights(agent: Agent):
    weights = []
    for model in ["cnn_encoder", "mlp_encoder", "mlp_policy"]:
        model = getattr(agent.policy, model)
        weights.append([val.cpu().numpy() for val in model.state_dict().values()])

    # Flatten the weights
    weights_flat = [val for model in weights for val in model]
    """ weights_lengeth = [len(model) for model in weights] #[8, 4, 4]
    print(f"Weights shape:")
    for isx, model in enumerate(weights):
        print(f"Shape for model: {["cnn_encoder", "mlp_encoder", "mlp_policy"][isx]}")
        for val in model:
            print("    ", val.shape) """
    return weights_flat

def set_weights(agent: Agent, parameters: list):

    # Set the weights of the model BCPolicy
    # unflatten the weights based on the [8, 4, 4] lengths
    parameters = [parameters[:8], parameters[8:12], parameters[12:]]

    """ # get curr_size
    weights = []
    # print("net attributes:", net.__dict__.keys())
    # print(f"State dict keys: {net.state_dict().keys()}")
    for model in ["cnn_encoder", "mlp_encoder", "mlp_policy"]:
        model = getattr(net, model)
        # weights[model] = {name: val.cpu().numpy() for name, val in model.state_dict().items()}
        weights.append([val.cpu().numpy() for name, val in model.state_dict().items()])
    for isx, model in enumerate(weights):
        print(f"Shape for model: {["cnn_encoder", "mlp_encoder", "mlp_policy"][isx]}")
        for val in model:
            print("    ", val.shape) """

    for idx, model in enumerate(["cnn_encoder", "mlp_encoder", "mlp_policy"]):
        model = getattr(agent.policy, model)
        params_dict = zip(model.state_dict().keys(), parameters[idx])
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
####################################
# This file is part of ELSA.
# 
# In this file, you need to implement an **agent** that will be used to interact with the environment. 
# The agent should be responsible for processing observations and selecting actions.
#
# **Agent Requirements:**
# The agent must be implemented as a class that inherits from `torch.nn.Module` and should include:
#
# - `__init__(self)`:  
#   - Define the architecture of your neural network here.  
#   - You can use convolutional layers for image processing and fully connected layers for decision-making.
#
# - `get_action(self, image, low_dim_state)`:  
#   - This method takes an image and a low-dimensional state as input and returns an action.
#   - You can apply post-processing (e.g., argmax for discrete actions or squashing for continuous actions).
#
#
# **Expected Input and Output for the get_action function:**
# - Inputs:
#   - `image`: A tensor representing the visual observation (e.g., RGB image).
#   - `low_dim_state`: A tensor containing additional state information.
# - Output:
#   - An action tensor that determines the agent's response to the given input. 
#    NOTE: The output action should be between [-1,1] ad it will be later denormalized to the action space of the environment.
#
# **Example Implementation:**
# Below is an example of a behavioral cloning model that combines a CNN encoder for image processing 
# and an MLP encoder for low-dimensional state inputs. It outputs an action based on the processed observations.
#
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Agent():
    def __init__(self, image_channels=3, low_dim_state_dim=10, action_dim=4, image_size=(64, 64)):
        # YOUR CODE HERE
        # Define the architecture of your neural network here
        # We provide an example by implementung a CNN policy we called BC Policy
        self.policy = BCPolicy(
        image_channels=image_channels,
        low_dim_state_dim=low_dim_state_dim,
        action_dim=action_dim,
        image_size=image_size
        )

    def train(self, ):
        self.policy.train()

    def eval(self, ):
        self.policy.eval()

    def get_action(self, image, low_dim_state):
        # Get the action from the policy
        return self.policy(image, low_dim_state) # Assuming the policy is a function
    
    def load_state_dict(self, state_dict,device=None):
        if device is None:
            self.policy.load_state_dict(torch.load(state_dict))
        else:
            self.policy.load_state_dict(torch.load(state_dict), map_location=torch.device(device))
        return self
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        return self



####################################################
# Your code here
# ...
#
#
#
######## EXAMPLE IMPLEMENTATION of CNN policy ########

class CNNEncoder(nn.Module):
    """CNN to encode the image observation with dynamic adjustment for fully connected layers."""
    def __init__(self, input_channels=3, output_dim=256, image_size=(64, 64)):
        """
        Args:
            input_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            output_dim (int): Size of the output embedding.
            image_size (tuple): Size of the input image (height, width).
        """
        super(CNNEncoder, self).__init__()
        self.image_size = image_size
        self.output_dim = output_dim

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Dynamically calculate the flattened size after convolutions
        self.flattened_size = self._calculate_flattened_size()

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, output_dim),
            nn.ReLU()
        )

    def _calculate_flattened_size(self):
        """Calculate the size of the flattened feature map after convolutions."""
        with torch.no_grad():
            # Create a dummy tensor with the given image size
            dummy_input = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
            dummy_output = self.cnn(dummy_input)  # Pass it through the CNN
            return int(torch.flatten(dummy_output, start_dim=1).size(1))  # Flatten and get the size

    def forward(self, x):
        x = self.cnn(x)  # Apply convolutions
        x = torch.flatten(x, start_dim=1)  # Flatten the feature map
        x = self.fc(x)  # Apply the fully connected layer
        return x

class MLPEncoder(nn.Module):
    """MLP to encode the low-dimensional state."""
    def __init__(self, input_dim, output_dim=128):
        super(MLPEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)
    

class BCPolicy(nn.Module):
    """Behavior Cloning model combining CNN and MLP encoders."""
    def __init__(self, image_channels=3, low_dim_state_dim=10, action_dim=4, image_size=(64, 64)):
        super(BCPolicy, self).__init__()
        self.cnn_encoder = CNNEncoder(input_channels=image_channels, output_dim=256, image_size=image_size)
        self.mlp_encoder = MLPEncoder(input_dim=low_dim_state_dim, output_dim=128)
        self.mlp_policy = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()  # Outputs bounded between -1 and 1
        )

    def forward(self, image, low_dim_state):
        img_embedding = self.cnn_encoder(image)
        state_embedding = self.mlp_encoder(low_dim_state)
        fused = torch.cat([img_embedding, state_embedding], dim=-1)
        return self.mlp_policy(fused)
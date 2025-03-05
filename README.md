# ELSA Robotics Benchmark: Federated Imitation Learning for Robotic Manipulation

![](https://elsa-ai.eu/wp-content/uploads/2024/05/banner_homepage_elsa_website_more_space_small.png)

This repository contains the code for the [ELSA Robotics Benchmark](https://benchmarks.elsa-ai.eu/?ch=5) task. For this challenge, a federated learning setup is provided to train a policy for a robotic arm to perform a series of tasks. The goal is to train a policy that can generalize to new tasks and different environment variations based on a given simulator based on the Colosseum Gym environment.

## Develop your own solution

Your submission should contain a custom implementation of the Agent Policy. A sample implementation is provided with its corresponding documentation in the file from our [GitHub repository](https://github.com/KTH-RPL/ELSA-Robotics-Challenge):

```bash
elsa_learning_agent/agent.py
```

You will have to complete the Agent class with your own implementation based on torch. The provided code is a simple example of an Imitation Learning agent. You can use this as a starting point to develop your own solution.

The agent takes as inputs the state of the environment, defined as two arrays with the following structure:

- `image`: RGB image of the front camera (128x128x3)
- `low_dim_state`: Low-dimensional variables of the environment (8x1)
  - `joint_positions`: Joint positions of the robotic arm (7 joint positions)
  - `gripper_open`: Gripper state (0: closed, 1: open) 

The agent should return a prediction as a list including the following keys:

- `actions`: Action to be executed by the robotic arm (7 joint velocities)
  - `joint_velocities`: Joint velocities of the robotic arm
  - `gripper_action`: Gripper action (0: close, 1: open)

For a more detailed explanation of the agent implementation, please refer to the provided code in the `elsa_learning_agent/agent.py` file.


## Create your own agent

### Download the required dependencies

First, download the docker image from the repository for the challenge using the following command:

```bash
docker pull santibou/elsa_robotics_challenge:latest
```

Download the code from our GitHub repository:

```bash
git clone https://github.com/KTH-RPL/ELSA-Robotics-Challenge.git
cd ELSA-Robotics-Challenge
```

Download the dataset from the [Dataverse](https://dataverse.harvard.edu/dataverse/elsa-robotics-challenge) and extract it into the `dataset` folder in the current repository root:

> To download the dataset, we have prepared a download script that uses the Hardvard Dataverse API to retrieve the files. For using it, you will need to provide the API token from your Dataverse Account. You can create this token by registering an account at [Hardvard Dataverse](https://dataverse.harvard.edu/). Once you have done it, access your account page and then click on the [API Token generation tab](https://dataverse.harvard.edu/dataverseuser.xhtml?selectTab=apiTokenTab). 
> Once you have the token, copy it into the `API_TOKEN` variable in the `download_dataset.py` file and run the script to download the dataset.
> ```bash
> python download_dataset.py --data_type training --task close_box
> ```
> 
> Here you can specify the task you want to download by changing the `task` parameter. The available tasks are: `slide_block_to_target`, `close_box`, `insert_onto_square_peg` and `scoop_with_spatula`.
>
> You can also specify the `data_type` which can be either `training`, `eval` or `test` to download the training, evaluation or test datasets respectively into their corresponding folders.
>
> Note: You can download a subset of the dataset by specifying the amount of environments to download using the flag `--num_envs`. For example, to download the first 5 environments of the `close_box` task, you can use the following command:
> ```bash
> python download_dataset.py --data_type training --task close_box --num_envs 5
> ```

After this steps, your project's folder structure should look like this:

```bash
ELSA-Robotics-Challenge/
│
├── datasets/
|   ├── training/
|   │   ├── close_box/
|   │   |   ├── close_box_fed.json
|   │   |   ├── close_box_fed.yaml
|   │   |   ├── env_0/
|   │   |   |   ├── episodes_observations.pkl.gz
|   │   |   |   ├── variation_descriptions.pkl
|   │   |   ├── ...
|   │   ├── slide_block_to_target/
|   │   ├── ...
|   │
|   ├── eval/
|   |   ├── ...
|   ├── test/
|       ├── ...
│
├── elsa_learning_agent/
│   ├── agent.py # Your agent implementation
│   ├── ...
│
├── federated_elsa_robotics/ # Main code for the federated learning training
│   ├── ...
│
├── dataset_config.py
├── pyproject.toml
├── ...
```

---

### Write the model for your agent

You can now start developing your own solution by modifying the `elsa_learning_agent/agent.py` file. You can use the provided code as a starting point to develop your own solution. Once the agent is implemented, you can start the simulation using flower.


### Train your agent

You can then run your simulation by starting the docker container from your current folder with the following command:

```bash
docker run -it --rm -v ./:/elsa_robotics_challenge/ --network=host --gpus=all --user $(id -u):$(id -g) --name elsa_robotics_challenge santibou/elsa_robotics_challenge:latest
```

Once the interactive shell opens, you can use `flwr run` to run a training run inside the docker container:

```bash
cd /elsa_robotics_challenge/
flwr run .
```

>#### Modify the run configuration
>
>In order to change the parameters of the federated learning, modify the `pyproject.toml` file. Some of the most relevant options you can change are:
>
>- `num-server-rounds`: Number of rounds for the federated learning
>- `local-epochs`: Number of epochs for each federated client's local training
>- `fraction-fit`: Fraction of clients that participate in each round (0.05 means 20 clients will participate if we set the number of nodes to 400)
>- `dataset-task`: Robot environment task to be used for training the policy
>- `train-split`: Fraction of the dataset to be used for training


### Evaluate your solution

Once the training is finished, the trained models will be located under the `model-checkpoints` folder, with a subfolder for each task. This model, saved for each round, can be then evaluated to obtain the specific metrics for the task. The evaluation time can be prolonged and depends on the amount of evaluation rounds set on the script (in our case, processing 30 rounds took around 1 hour of compute time on 64 CPUs and a NVIDIA A100 GPU).

You can evaluate your solution by running the following command with the `--plotting` flag to visualize the results (you can adjust the other parameters based on the training configuration used in the previous step):

```bash
python -m federated_elsa_robotics.eval_model --task slide_block_to_target --local_epochs 50 --rounds_to_evaluate 30 --fraction_fit 0.05 --train_test_split 1.0 --plotting
```

The evaluation script will output the following metrics:

- `Success Rate`: Percentage of successful episodes on the live dataset evaluation
- `RMSE loss`: Root Mean Squared Error loss computed from the comparison of the model policies and the dataset ground truth actions.

## Results Baseline

For this benchmark, we also provide a baseline of results obtained by the `BCPolicy` model provided in the `elsa_learning_agent/agent.py` file.

The baseline configuration is the following:

- `dataset-task`: close_box
- `num-server-rounds`: 30
- `local-epochs`: 50
- `fraction-fit`: 0.05 (20 clients)
- `train-split`: 0.9


The testing on the evaluation dataset for the task with the best accuracy, `close_box`, results in the following model metrics for the most successful round:

| Round | Mean Loss | Std Loss | Mean Success Rate | Std Success Rate |
|:-----:|:---------:|:--------:|:-----------------:|:----------------:|
| 28    | 0.006577      | 0.002189     | 56.00%            | 49.64%           |



## Final Evaluation and Submission

Once you have your final trained policy agent, you can extract the observations and ground truth actions for the final test dataset. For doing so, firstly you need to set up the paths of your best trained models according to the results from the previous evaluation, you can do so modifying the `model_paths` dictionary at `elsa_learning_agent/scripts/policy_evaluation_rrc.py` main function.

You can then run the following script to obtain the resulting predictions file (`results/predictions.json`)  with the observations and ground truth actions that can be submitted to the [ELSA Robotics Benchmark](https://benchmarks.elsa-ai.eu/?ch=5) website:

```bash
python -m elsa_learning_agent.scripts.policy_evaluation_rrc
```

### Submission

To submit your final solution, you will have to upload the `results/actions.json` file generated by the previous script to the [ELSA Robotics Benchmark](https://benchmarks.elsa-ai.eu/?ch=5) website.


## Contact

If you have any questions or need help with the setup, feel free to contact us at:

- [Miguel Serras Vasco](mailto:miguelsv@kth.se)
- [Santiago Bou Betran](mailto:sbb@kth.se)
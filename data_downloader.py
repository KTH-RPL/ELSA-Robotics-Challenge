import json
import os
import argparse
from urllib import request

NUM_PROCESSES = 64
# Upload a folder by chunking it into multiple uploads
DV_URL = "https://dataverse.harvard.edu/"
# INTRODUCE HERE YOU API TOKEN FROM DATAVERSE HARDVARD
API_TOKEN = "API_TOKEN"

def download_envs(path, task, envs, files, PID):
    for env in envs:
        file_ids = [file for file in files if f"{path}/{env}" in file["directoryLabel"]]
        print(f"Downloading {env}")
        os.makedirs(f"./datasets/{path}/{env}", exist_ok=True)
        for file_id in file_ids:
            if "episodes_observations.pkl.gz" in file_id["dataFile"]["filename"] and not os.path.exists(f"./datasets/{path}/{env}/episodes_observations.pkl.gz"):
                obs_id = file_id["dataFile"]["id"]
                request.urlretrieve(f"https://dataverse.harvard.edu/api/access/datafile/persistentId/{obs_id}?persistentId={PID}", f"./datasets/{path}/{env}/episodes_observations.pkl.gz")
            elif "variation_descriptions.pkl" in file_id["dataFile"]["filename"] and not os.path.exists(f"./datasets/{path}/{env}/variation_descriptions.pkl"):
                var_id = file_id["dataFile"]["id"]
                request.urlretrieve(f"https://dataverse.harvard.edu/api/access/datafile/:persistentId/{var_id}?persistentId={PID}", f"./datasets/{path}/{env}/variation_descriptions.pkl")
            
    if not os.path.exists(f"./datasets/{path}/{task}_fed.json") or not os.path.exists(f"./datasets/{path}/{task}_fed.yaml"):
        file_id = [file for file in files if f"{task}_fed.json" in file["label"]][0]["dataFile"]["id"]
        request.urlretrieve(f"https://dataverse.harvard.edu/api/access/datafile/persistentId/{file_id}?persistentId={PID}", f"./datasets/{path}/{task}_fed.json")

    if not os.path.exists(f"./datasets/{path}/{task}_fed.yaml"):
        file_id = [file for file in files if f"{task}_fed.yaml" in file["label"]][0]["dataFile"]["id"]
        request.urlretrieve(f"https://dataverse.harvard.edu/api/access/datafile/persistentId/{file_id}?persistentId={PID}", f"./datasets/{path}/{task}_fed.yaml")

def download(envs_per_chunk, task, path, PID, data_type, num_envs):
    print(f"Downloading to {path}")

    start_env = 0
    if data_type == "eval":
        start_env = 400
    
    if data_type == "test":
        len_dataset = 1
    elif num_envs == 400:
        if data_type == "training":
            len_dataset = 400
        elif data_type == "eval":
            len_dataset = 50
    else:
        if data_type == "training" and num_envs <= 400:
            len_dataset = num_envs
        elif data_type == "eval" and num_envs <= 50:
            len_dataset = num_envs
        else:
            raise ValueError("Number of environments should be maximum 400 for training and 50 for eval")
        
    envs = ["env_" + str(i) for i in range(start_env, len_dataset+start_env)]
    envs = sorted(envs, key=lambda x: int(x.split("_")[-1]))
    chunk_envs = [envs[i:i+envs_per_chunk] for i in range(0, len(envs), envs_per_chunk)]

    files = json.loads(request.urlopen(f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId={PID}").read().decode("utf-8"))["data"]["latestVersion"]["files"]
    files = [file for file in files if task in file["directoryLabel"]]

    for i, envs in enumerate(chunk_envs):
        print(f"\n --> Downloading envs {i*envs_per_chunk} to {(i+1)*envs_per_chunk}")
        download_envs(path, task, envs, files, PID)
    

def install_API_token():
    opener = request.build_opener()
    opener.addheaders = [("X-Dataverse-key", API_TOKEN)]
    request.install_opener(opener)

def main(args):
    envs_per_chunk = 5

    if args.data_type == "eval":
        PID = "doi:10.7910/DVN/DOZY6N"
    elif args.data_type == "test":
        PID = "doi:10.7910/DVN/OVVN2E"
    elif args.data_type == "training":
        if args.task == "slide_block_to_target":
            PID = "doi:10.7910/DVN/XEAYPQ"
        elif args.task == "close_box":
            PID = "doi:10.7910/DVN/QPUOJH"
        elif args.task == "insert_onto_square_peg":
            PID = "doi:10.7910/DVN/PSPXJK"
        elif args.task == "scoop_with_spatula":
            PID = "doi:10.7910/DVN/EPU7UW"

    # envs_per_chunk = args.envs_per_chunk
    data_type = args.data_type
    task = args.task
    path = f"{data_type}/{task}"

    install_API_token()
    download(envs_per_chunk, task, path, PID, args.data_type, args.num_envs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="training", choices=["training", "eval", "test"])
    parser.add_argument("--task", type=str, default="close_box", choices=["slide_block_to_target", "close_box", "insert_onto_square_peg", "scoop_with_spatula"])
    parser.add_argument("--num_envs", type=int, default=400)
    args = parser.parse_args()
    main(args)
import argparse
import json
import random
import numpy as np
import pandas as pd
import json
from datetime import date, timedelta, datetime
import ast
import itertools
import torch

from data_utils import load_raw_data, save_data, load_clean_data
from patient_similarity import jaccard_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw single-cell data.")

    # Task
    parser.add_argument("--raw", type=bool, default=False, help="Process raw data")
    parser.add_argument("--split", type=str, default=None, choices=["state", "random", "counterfactual"], help="Split data")

    # Input/Output
    parser.add_argument("--data_type", type=str, default=None, choices=["WaddingtonOT"], help="Data type")
    parser.add_argument("--data_dir", type=str, default="../data/", help="Directory to raw cell data")
    parser.add_argument('--save_prefix', type=str, default='../data/', help='Prefix of all saved files')

    args = parser.parse_args()
    return args


def convert_step2date(step):
    startdate = datetime.fromisoformat("2000-01-01T00:00:00")
    step = timedelta(hours = float(step), seconds = 1)
    return startdate + step


def chunk(state, df, keys, size):
    eps = 10e-8
    chunk_trajs = []
    batch_id = 0
    
    # Padding
    if len(keys) % 2 == 1: keys.append("pad")

    # Sliding window if size = 2
    sliding_window = (size == 2)
    print("Sliding window or disjoint chunks:", size, sliding_window)
    if sliding_window: iterator = range(0, len(df) - 1)
    else: iterator = range(0, len(df), size)
    
    # Do chunking
    for i in iterator:
        subset_df = df.iloc[i:i+size, :]
        assert subset_df.shape[0] == size, (subset_df.shape, size)
        subject_id = state + "_batch" + str(batch_id)
        for step, vec, act_i in zip(subset_df.index.tolist(), subset_df["states_readable"].tolist(), subset_df["actions_int"].tolist()):
            vec = [int(i) + eps for i in list(vec)]
            
            # Padding
            if len(vec) % 2 == 1: vec.append(0)
            assert len(vec) == len(keys)

            step = convert_step2date(step)
            action = float(act_i) #keys[int(float(act_i))]
            if np.isnan(action): action = float(-1.0)
            vec_df = pd.DataFrame({"subject_id_str": [subject_id] * len(keys),
                                   "label": keys, "itemid": keys,
                                   "valuenum": vec,
                                   "action": [action] * len(keys),
                                   "charttime": [step] * len(keys)})
            chunk_trajs.append(vec_df)
        batch_id += 1
    chunk_trajs = pd.concat(chunk_trajs, ignore_index = True)
    subject_id_map = {k: i for i, k in enumerate(chunk_trajs["subject_id_str"].unique())}
    chunk_trajs["subject_id"] = chunk_trajs["subject_id_str"].map(subject_id_map)
    chunk_trajs["charttime"] = chunk_trajs["charttime"].astype(str)# + " 00:00:00"
    print(chunk_trajs)
    return chunk_trajs


def process_raw_data(states_data, data_type):
    
    if data_type == "WaddingtonOT":

        if "counterfactual" not in states_data:
            states_mapping = {s: (i + 1) * 1000000 for i, s in enumerate(['start1', 'start2', 'start3'])}
            print(states_mapping)
        else:
            states_mapping = {s: (i + 1) * 1000000 for i, s in enumerate(['original', 'counterfactual'])}
            print(states_mapping)

        all_trajs = []
        unique_actions = []
        observation_dims = []
        traj_lengths = []
        action_embs_dict = dict()
        for state, df in states_data.items():

            print("Max time steps:", df["timestep"].astype(float).max())
            
            groups = df["group"].unique()
            ungrouped_groups = [g for g in groups if g.startswith("Ungrouped")]
            celltype_groups = [g for g in groups if not g.startswith("Ungrouped")]
            print("There are %d ungrouped groups:" % len(ungrouped_groups), ungrouped_groups)
            print("There are %d celltype groups:" % len(celltype_groups), celltype_groups)
            
            state_trajs = []
            print("There are %d trajectories in state %s" % (len(df["trajectory"].unique()), state))

            action_embs_df = df[["action", "actions"]].drop_duplicates()
            traj_actions = action_embs_df["action"].apply(ast.literal_eval).tolist()            
            action_embs = action_embs_df["actions"].apply(ast.literal_eval).tolist()
            action_embs = torch.tensor(action_embs)
            action_embs = {k[0]: action_embs[i, :] for i, k in enumerate(traj_actions) if len(k) > 0} # Assume single TF action
            if "None" not in action_embs: action_embs["None"] = torch.zeros(len(list(action_embs.values())[0])) # No action embedding
            for a, e in action_embs.items():
                if a in action_embs_dict: assert torch.equal(e, action_embs_dict[a])
            action_embs_dict.update(action_embs)
            print("Updated action_embs_dict", len(action_embs_dict))

            for traj in df["trajectory"].unique():
                traj_df = df[df["trajectory"] == traj].reset_index()

                observations = traj_df["observations"].apply(ast.literal_eval).tolist()
                observations = np.array(observations)

                # Add padding
                if observations.shape[1] % 2 == 1:
                    observations = np.concatenate([observations, np.zeros((observations.shape[0], 1))], axis = 1)
                assert observations.shape[1] % 2 == 0

                traj_actions = traj_df["action"].apply(ast.literal_eval).tolist()
                unique_actions.extend(traj_actions)
                extend_traj_actions = []
                for a in traj_actions:
                    assert len(a) <= 1
                    if len(a) == 0:
                        extend_traj_actions.extend(["None"] * observations.shape[1])
                    else:
                        extend_traj_actions.extend([a[0]] * observations.shape[1])
                
                traj_lengths.append(observations.shape[0])
                observation_dims.append(observations.shape[1])
                assert traj_df.shape[0] == observations.shape[0]
                
                # Convert time step
                steps = []
                for t in traj_df["timestep"].tolist():
                    steps.extend([convert_step2date(t * 10)] * observations.shape[1]) # Multiply each time step by 10 to remove decimal
                
                # Reshape observations and labels
                observations_flat = observations.flatten()
                labels = np.tile(np.arange(observations.shape[1]), observations.shape[0])
                assert len(observations_flat) == len(labels)

                assert len(extend_traj_actions) == len(observations_flat), (len(extend_traj_actions), len(observations_flat))
                assert len(steps) == len(observations_flat), (len(steps), len(observations_flat))
                
                # Combine
                agg_df = pd.DataFrame({"subject_id_str": [state + "_" + str(traj)] * len(observations_flat),
                                       "subject_id": [states_mapping[state] + traj] * len(observations_flat),
                                       "label": labels, "itemid": labels,
                                       "valuenum": observations_flat,
                                       "action": extend_traj_actions,
                                       "charttime": steps})
                state_trajs.append(agg_df)
                
            state_trajs = pd.concat(state_trajs, ignore_index = True)
            print(state_trajs)
            all_trajs.append(state_trajs)
        
        all_trajs = pd.concat(all_trajs, ignore_index = True)
        print(all_trajs)

        # Tokenize action
        unique_actions = set(list(itertools.chain.from_iterable(unique_actions)))
        assert len(set(unique_actions).intersection(set(list(action_embs_dict.keys())))) == len(unique_actions)
        action_tokenize = {a: float(i + 1) for i, a in enumerate(sorted(unique_actions))}
        action_tokenize["None"] = float(-1.0)
        print(action_tokenize)
        all_trajs["action_idx"] = all_trajs["action"].map(action_tokenize)

        # Final touches
        all_trajs["charttime"] = all_trajs["charttime"].astype(str)
        
        print("Average length of trajectories:", np.mean(traj_lengths), np.std(traj_lengths))
        print("Average dims of observations:", np.mean(observation_dims), np.std(observation_dims))
        print("There are %d unique actions" % len(unique_actions))

        return all_trajs, action_embs_dict
    
    else:
        raise NotImplementedError
    
    return None


def split_cells(data_type, data, split):
    if split == "state":
        
        states = set([i.split("_")[0] for i in data["subject_id_str"].tolist()])
        print(states)

        if data_type == "WaddingtonOT":
            train_prefix = ["start1"]
            val_prefix = ["start3"]
            test_prefix = ["start2"] # Supposedly a very weird cluster
        else:
            raise NotImplementedError

        states_mapping = {k: str(v) for k, v in zip(data["subject_id_str"].tolist(), data["subject_id"].tolist())}

        train = []
        val = []
        test = []
        for i, i_id in states_mapping.items():
            i_prefix = i.split("_")[0]
            if i_prefix in train_prefix: train.append(i_id)
            elif i_prefix in val_prefix: val.append(i_id)
            else: test.append(i_id)

        assert len(data["subject_id"].unique()) == len(train + val + test), (data["subject_id"].unique(), train + val + test)
        print("Number of unique subjects:", len(data["subject_id"].unique()))

        return {"train": ";".join(train), "val": ";".join(val), "test": ";".join(test)}
    
    elif split == "random":

        unique_traj = list(data["subject_id"].astype(str).unique())
        train = random.sample(unique_traj, int(len(unique_traj) * 0.8))
        remain = set(unique_traj).difference(set(train))
        val = random.sample(remain, int(len(remain) * 0.2))
        test = set(remain).difference(set(val))
        assert len(set(train).intersection(val)) == 0
        assert len(set(val).intersection(test)) == 0
        assert len(set(train).intersection(test)) == 0
        print("Number of train:", len(train))
        print("Number of val:", len(val))
        print("Number of test:", len(test))

        return {"train": ";".join(train), "val": ";".join(list(val)), "test": ";".join(list(test))}

    elif split == "counterfactual":

        train_prefix = ["original"]
        test_prefix = ["counterfactual"]
        
        states_mapping = {k: str(v) for k, v in zip(data["subject_id_str"].tolist(), data["subject_id"].tolist())}

        train = []
        test = []
        for i, i_id in states_mapping.items():
            i_prefix = i.split("_")[0]
            if i_prefix in train_prefix: train.append(i_id)
            else: test.append(i_id)
        val = random.sample(train, int(len(train) * 0.2))
        train = list(set(train).difference(set(val)))

        assert len(set(train).intersection(set(val))) == 0
        assert len(set(train).intersection(set(test))) == 0
        assert len(set(val).intersection(set(test))) == 0
        print("Train:", len(train), "Validation:", len(val), "Test:", len(test))
        assert len(data["subject_id"].unique()) == len(train + val + test), (data["subject_id"].unique(), train + val + test)
        print("Number of unique subjects:", len(data["subject_id"].unique()))

        return {"train": ";".join(train), "val": ";".join(val), "test": ";".join(test)}

    else:
        raise NotImplementedError


def main():
    args = parse_args()
    
    if args.raw:
        states_data = load_raw_data(args.data_dir, args.data_type)
        clean_data, action_embs_dict = process_raw_data(states_data, args.data_type)
        save_data(args.save_prefix + "filtered_data.csv", clean_data)
        save_data(args.save_prefix + "action_embs_dict.pth", action_embs_dict)
    else:
        print("Reading in cleaned data...")
        clean_data = load_clean_data(args.save_prefix + "filtered_data.csv")
        print(clean_data)

    if args.split is not None:
        if args.split == "counterfactual": outfile = args.save_prefix + "filtered_data_split=state.json"
        else: outfile = args.save_prefix + ("filtered_data_split=%s.json" % args.split)
        split_data = split_cells(args.data_type, clean_data, args.split)
        save_data(outfile, split_data)


if __name__ == "__main__":
    main()

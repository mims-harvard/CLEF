import os
import argparse
import json
import random
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import math
import copy
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../') # Add our own scripts
from project_config import PROJECT_DIR

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


DATE_PADDING = "0001-01-01 00:00:00"


def parse_args():
  parser = argparse.ArgumentParser(description="Train model.")

  # General model architecture
  parser.add_argument("--seq_encoder", type=str, default="transformer", choices = ["transformer", "xlstm", "moment"], help="Encoder for sequential data")
  parser.add_argument("--loss", type=str, default="huber", help="Loss function")
  parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
  parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
  parser.add_argument("--nfeat", type=int, default=0, help="Number of input features")
  parser.add_argument("--nhead", type=int, default=1, help="Number of heads")
  parser.add_argument("--nlayers", type=int, default=0, help="Number of layers")
  parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate")
  parser.add_argument("--conv_sz", type=int, default=0, help="Conv1d kernel size (only relevant to xLSTM)")
  parser.add_argument("--qkv_proj_sz", type=int, default=0, help="Projection for qkv (only relevant to xLSTM)")
  parser.add_argument("--batch_sz", type=int, default=512, help="Batch size")
  parser.add_argument("--temporal_len", type=int, default=8, help="Number of historical time steps to provide as input")
  parser.add_argument("--condition", default=False, action="store_true", help="Condition (not time)")
  parser.add_argument("--best", default=False, action="store_true", help="Run predefined best hyperparameters based on dataset and encoder type")
  
  # Concept-based learning
  parser.add_argument("--nconcepts", type=int, default=0, help="Number of concepts")
  parser.add_argument("--apply_ct_proj", default=False, action="store_true", help="Use concept projection layer")
  
  # Baselines/Ablations
  parser.add_argument("--concept_ones", default=False, action="store_true", help="Baseline: next = prev")

  # Input/Output
  parser.add_argument("--data_type", type=str, default="patient", choices=["patient", "cell"], help="Split data")
  parser.add_argument("--save_concepts", default=False, action="store_true", help="Track/save concepts")
  parser.add_argument("--save_preds", default=False, action="store_true", help="Save predictions")
  parser.add_argument("--data_dir", type=str, default=("%sdata/MIMIC-IV/" % PROJECT_DIR), help="Directory to processed EHR data")
  parser.add_argument("--split", type=str, default="patient", choices=["patient", "time", "state"], help="Split data")
  parser.add_argument("--spectra", type=str, default="", help="SPECTRA parameter (or empty string if not using SPECTRA)")
  parser.add_argument("--counterfactual", default=False, action="store_true", help="Generate counterfactual predictions")
  parser.add_argument("--edit", default=False, action="store_true", help="Generate edit counterfactual predictions")
  parser.add_argument("--time_skip", default=False, action="store_true", help="Predict immediate next time step t+1 (i.e., no skip; False) or future t+T (i.e., skip; True)")
  parser.add_argument("--output_dir", type=str, default=("%sresults/" % PROJECT_DIR), help="Output directory")
  parser.add_argument("--save_prefix", type=str, default="", help="Prefix of all saved files")
  
  # Training/inference
  parser.add_argument("--resume", type=str, default="", help="Path to best checkpoint")
  parser.add_argument("--inference", default=False, action="store_true", help="Run inference without training")
  parser.add_argument("--seed", type=int, default=1, help="Seed")

  args = parser.parse_args()
  return args


def set_seed(seed):
  print("Setting seed:", seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) 
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True


def get_hparams(args):

  hparams = {
              "epochs": args.epochs,

              "seq_encoder": args.seq_encoder,
              "loss": args.loss,
              "temporal_len": args.temporal_len,
              "condition": args.condition, # Boolean

              "lr": args.lr,
              "nhead": args.nhead,
              "conv_sz": args.conv_sz,
              "qkv_proj_sz": args.qkv_proj_sz,
              "nfeat": args.nfeat,
              "apply_ct_proj": args.apply_ct_proj,
              "nconcepts": args.nconcepts,
              "nlayers": args.nlayers,
              "dropout": args.dropout,
              
              "batch_sz": args.batch_sz,
              "concept_ones": args.concept_ones,
              "split": args.split,
              "spectra": args.spectra,
              "counterfactual": args.counterfactual,
              "edit": args.edit
            }
  
  # Run predefined best hyperparameters based on dataset and encoder type
  if args.best:
    print("Selecting predefined best hyperparameters based on dataset and encoder type...")
    print("Warning: Model architecture related arguments provided in your command may be overwritten.")

    # Transformer encoder
    if hparams["seq_encoder"] == "transformer":
      if "MIMIC" in args.data_dir:
        hparams.update({
                        "nhead": 4,
                        "nlayers": 8,
                        "nfeat": 16,
                        "dropout": 0.6,
                        "lr": 0.0001
                      })
      elif "eICU" in args.data_dir:
        hparams.update({
                        "nhead": 6,
                        "nlayers": 8,
                        "nfeat": 18,
                        "dropout": 0.6,
                        "lr": 0.0001
                      })
      elif "UCE" in args.data_dir:
        hparams.update({
                        "nhead": 8,
                        "nlayers": 4,
                        "nfeat": 1280,
                        "dropout": 0.6,
                        "lr": 0.00001
                      })
      elif "HVG" in args.data_dir:
        hparams.update({
                        "nhead": 8,
                        "nlayers": 4,
                        "nfeat": 1480,
                        "dropout": 0.6,
                        "lr": 0.00001
                      })
      else: raise NotImplementedError

    # Pretrained MOMENT encoder
    elif hparams["seq_encoder"] == "moment":
      if "MIMIC" in args.data_dir:
        hparams.update({
                        "nfeat": 16,
                        "dropout": 0.6,
                        "lr": 0.0001
                      })
      elif "eICU" in args.data_dir:
        hparams.update({
                        "nfeat": 18,
                        "dropout": 0.6,
                        "lr": 0.0001
                      })
      elif "UCE" in args.data_dir:
        hparams.update({
                        "nfeat": 1280,
                        "dropout": 0.6,
                        "lr": 0.00001
                      })
      elif "HVG" in args.data_dir:
        hparams.update({
                        "nfeat": 1480,
                        "dropout": 0.6,
                        "lr": 0.00001
                      })
      else: raise NotImplementedError

    # xLSTM encoder
    elif hparams["seq_encoder"] == "xlstm":
      if "MIMIC" in args.data_dir:
        hparams.update({
                        "nhead": 4,
                        "nlayers": 8,
                        "nfeat": 16,
                        "conv_sz": 4,
                        "qkv_proj_sz": 4,
                        "dropout": 0.6,
                        "lr": 0.0001
                      })
      elif "eICU" in args.data_dir:
        hparams.update({
                        "nhead": 2,
                        "nlayers": 8,
                        "nfeat": 18,
                        "conv_sz": 4,
                        "qkv_proj_sz": 4,
                        "dropout": 0.6,
                        "lr": 0.0001
                      })
      elif "UCE" in args.data_dir:
        hparams.update({
                        "nhead": 8,
                        "nlayers": 4,
                        "nfeat": 1280,
                        "conv_sz": 5,
                        "qkv_proj_sz": 8,
                        "dropout": 0.6,
                        "lr": 0.00001
                      })
      elif "HVG" in args.data_dir:
        hparams.update({
                        "nhead": 8,
                        "nlayers": 4,
                        "nfeat": 1480,
                        "conv_sz": 4,
                        "qkv_proj_sz": 8,
                        "dropout": 0.6,
                        "lr": 0.00001
                      })
      else: raise NotImplementedError
    
    else: raise NotImplementedError
  return hparams


def check_do_train(args, hparams):
  if args.inference or args.counterfactual or args.edit: return False
  if hparams["concept_ones"]: return False
  return True


def RMSE(pred, true):
  return torch.sqrt(torch.mean((pred - true) ** 2))


def MAPE(pred, true):
  return torch.mean(torch.abs((pred - true) / true))


def get_loss_func(loss_type):
  if loss_type == "mse":
    loss = nn.MSELoss()
    con_loss = nn.MSELoss()
  elif loss_type == "rmse":
    loss = RMSE()
    con_loss = RMSE()
  elif loss_type == "mae":
    loss = nn.L1Loss()
    con_loss = nn.L1Loss()
  elif loss_type == "huber":
    loss = nn.HuberLoss()
    con_loss = nn.HuberLoss()
  elif loss_type == "mape":
    loss = MAPE()
    con_loss = MAPE()
  else:
    raise NotImplementedError
  return loss, con_loss


def calc_metrics(true, pred, metrics = {}):
  r2 = r2_score(true, pred)
  rmse = root_mean_squared_error(true, pred)
  mae = mean_absolute_error(true, pred)
  mape = mean_absolute_percentage_error(true, pred)
  sim = cosine_similarity(true.reshape(1, -1), pred.reshape(1, -1))
  if len(metrics) != 0:
    metrics["r2"].append(r2)
    metrics["rmse"].append(rmse)
    metrics["mae"].append(mae)
    metrics["mape"].append(mape)
    metrics["sim"].append(sim)
  return r2, rmse, mae, mape, sim, metrics


def get_single_dense_patient_data(data, p_id, time = None):
  p_data = data[data["subject_id"] == p_id]
  p_data = p_data.pivot(index="charttime", columns="itemid", values="valuenum")
  if time is not None:
    p_data = p_data.iloc[time[p_id], :]
  return p_data


def get_action_emb(a, action_embs):
  if len(action_embs) == 0: return torch.full((1,), float('nan'))
  if type(a) != str or a not in action_embs: emb = action_embs["None"]
  elif ";" in a: emb = torch.stack([action_embs[a_i] for a_i in a.split(";")]).mean()
  else: emb = action_embs[a]
  return emb


MAX_TIME_CUTOFF = 20 # Save training time by setting cutoff of time
#MAX_TIME_CUTOFF = 23
def get_dense_subject_data(data, p_id, split, time = None, action_embs = dict(), apply_max_time_cutoff = True, time_skip = False):
  
  def select_indices(group, max_index, shuffle = False):
    start_idx = 0
    if group.shape[0] < max_index:
      padding_df = pd.DataFrame(np.nan, index=pd.MultiIndex.from_tuples([(("pad%d" % i), str(get_time(DATE_PADDING)), -1) for i in range(max_index - group.shape[0])]), columns=group.columns)
      group = pd.concat([group, padding_df])
      assert max_index == group.shape[0]
    if group.shape[0] > max_index:
      if shuffle: start_idx = np.random.randint(0, group.shape[0] - max_index + 1) # Randomly sample subset of consecutive rows
    if group.shape[1] % 2 == 1:
      group["pad"] = np.nan
      assert group.shape[1] % 2 == 0
    return group.iloc[start_idx:(start_idx + max_index), :]

  p_data = data[data["subject_id"].isin(p_id.tolist())]
  
  if time is not None: max_time = max(list({k: len(v) for k, v in time.items()}.values()))
  else: max_time = p_data[["subject_id", "charttime"]].drop_duplicates().groupby("subject_id").count().max().to_numpy()[0]
  if apply_max_time_cutoff:
    max_time = min(MAX_TIME_CUTOFF, max_time) # Reduce training time
    do_shuffle = True
  else: # Reduce inference time
    if split == "time" or time_skip:
      max_time = min(MAX_TIME_CUTOFF, max_time)
    else:
      max_time = min(MAX_TIME_CUTOFF * 10, max_time)
    do_shuffle = False

  # Pivot dataframe
  p_data = p_data.pivot(index=["subject_id", "charttime", "action"], columns="itemid", values="valuenum")
  
  # For each subject, extract max_time rows (+ padding if necessary)
  p_data = p_data.groupby("subject_id").apply(lambda x: select_indices(x, max_time, do_shuffle))

  # Tensor
  p_x = torch.tensor(p_data.to_numpy().reshape(len(p_id), max_time, p_data.shape[1])) # Batch x Visits x Labs
  p_dates = np.array([i[2] for i in p_data.index.tolist()]).reshape(len(p_id), max_time) # Batch x Visits
  if len(action_embs) > 0: action_emb_sz = len(list(action_embs.values())[0])
  else: action_emb_sz = 1
  p_actions = torch.stack([get_action_emb(i[3], action_embs) for i in p_data.index.tolist()]).reshape(len(p_id), max_time, action_emb_sz) # Batch x Visits x Embedding

  return p_x, p_dates, p_actions, p_data


def get_time(s):
  time = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
  assert type(time) == datetime
  return time


def get_split_data(all_data, get_type, split_dict):
  if get_type == "patient" or get_type == "state":
    patient_ids = [int(p) for p in split_dict.split(";")]
    subset_data = all_data[all_data["subject_id"].isin(patient_ids)]
    time_idx = None

  elif get_type == "time":
    patient_ids = [int(p) for p in list(split_dict.keys())]
    subset_data = all_data[all_data["subject_id"].isin(patient_ids)]
    time_idx = {k: [int(i) for i in v.split(";")] for k, v in split_dict.items()}

  else:
    raise NotImplementedError
  assert len(subset_data) > 0, len(subset_data)
  return [subset_data, time_idx]


def load_split_data(data_dir, hparams, data_type):

  if hparams["counterfactual"]:
    data_prefix = "counterfactual_data"
    split_prefix = None
  else:
    if data_type == "patient":
      data_prefix = "filtered_lab_code_data"
      split_prefix = "filtered_lab_data"
    elif data_type == "cell":
      data_prefix = "filtered_data"
      split_prefix = "filtered_data"
    else:
      raise NotImplementedError

  # Load processed data
  data = pd.read_csv(data_dir + data_prefix + ".csv", sep = "\t") # filtered_lab_data
  print(data)

  # Load condition embeddings
  if os.path.exists(data_dir + "action_embs_dict.pth"):
    action_embs_dict = torch.load(data_dir + "action_embs_dict.pth", map_location=torch.device('cpu'))
    hparams["condition_sz"] = len(list(action_embs_dict.values())[0])
    hparams["condition"] = True
    print("Action emb:", len(action_embs_dict), hparams["condition_sz"])
  else:
    hparams["condition_sz"] = 1
    if data_type == "patient" and "action" not in data.columns: data["action"] = -1

  if hparams["counterfactual"] or hparams["edit"]:

    # Get data splits
    if hparams["counterfactual"]:
      data["cf_type"] = data["cf_type"].fillna("original")
    else:
      # Load matched data
      matched_pt_f = data_dir + "matched_T1D_patients.json"
      print("LOADING MATCHED PATIENTS FROM %s" % matched_pt_f)
      matches = json.load(open(matched_pt_f, "r"))
      print("Number of matches:", len(matches))
      data = data[data["subject_id"].astype(str).isin(list(matches.keys()))]
  
    train_loader = [[], None]
    val_loader = [[], None]
    test_loader = [data, None]

    # Get maximum visits (ntoken)
    if "MIMIC" in data_dir: hparams["ntoken"] = 949
    elif "eICU" in data_dir: hparams["ntoken"] = 858
    elif "HVG" in data_dir: hparams["ntoken"] = 37
    elif "UCE" in data_dir: hparams["ntoken"] = 37
    else: raise NotImplementedError

  else:
    # Get data splits
    if hparams["spectra"] == "":
      split_f = data_dir + ("%s_split=%s.json" % (split_prefix, hparams["split"]))
    else:
      split_f = data_dir + ("SPECTRA/SP_%s_0/split_dict.json" % hparams["spectra"])
    split_dict = json.load(open(split_f, "r"))
    train_loader = get_split_data(data, hparams["split"], split_dict["train"])
    val_loader = get_split_data(data, hparams["split"], split_dict["val"])
    test_loader = get_split_data(data, hparams["split"], split_dict["test"])

    # Get maximum visits (ntoken)
    patient_visit_counts = data[["subject_id", "charttime"]].drop_duplicates().groupby("subject_id").count()
    hparams["ntoken"] = patient_visit_counts["charttime"].max()

  return data, action_embs_dict, train_loader, val_loader, test_loader, hparams


def create_look_ahead_mask(size):
  mask = torch.triu(torch.ones(size, size), 1)
  return mask


def convert_dates(d, device):
  if d.ndim == 1: d = d.reshape(len(d), 1)
  num_patients, num_visits = d.shape

  d_converted = {"year": [], "month": [], "day": [], "time": []}
  for d_i in d.flatten():
    assert type(d_i) == datetime
    d_converted["year"].append(d_i.year)
    d_converted["month"].append(d_i.month)
    d_converted["day"].append(d_i.day)
    d_converted["time"].append(d_i.hour)
  d_converted["year"] = torch.tensor(d_converted["year"]).long().reshape(num_patients, num_visits).to(device)
  d_converted["month"] = torch.tensor(d_converted["month"]).long().reshape(num_patients, num_visits).to(device)
  d_converted["day"] = torch.tensor(d_converted["day"]).long().reshape(num_patients, num_visits).to(device)
  d_converted["time"] = torch.tensor(d_converted["time"]).long().reshape(num_patients, num_visits).to(device)

  return d_converted


def tensor_dcn(x):
  return x.detach().cpu().numpy()


def calc_entropy(vec):
  if len(vec) == 0: return np.nan
  vec = torch.abs(vec)
  entr = entropy(tensor_dcn(torch.div(vec, torch.sum(vec, 1).unsqueeze(-1).tile((1, vec.shape[1])))))
  return np.nanmean(entr)


def get_patient_tensor(x, device):
  x_mask = torch.isnan(x).bool().to(device) # 1 = padding
  x = torch.nan_to_num(x, 0).float().to(device)
  return x, x_mask
  

def get_dates_mask(dates):
  mask = torch.tensor(dates == datetime.strptime(DATE_PADDING, '%Y-%m-%d %H:%M:%S'))
  if mask.dim() == 1: mask = mask.unsqueeze(-1)
  return mask


def agg_mask(x_mask, dates, device):
  dates_mask = get_dates_mask(dates).unsqueeze(-1).to(device)
  return torch.logical_or(x_mask, dates_mask)


def load_checkpoint(save_model, model, optimizer):
  checkpoint = torch.load(save_model, map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  return model, optimizer


def iterate_updated_pids(save_dict, p_ids_updated, x, mask):
  assert len(p_ids_updated) == x.shape[0], (len(p_ids_updated), x.shape)
  assert len(mask) == len(p_ids_updated)
  num_updated = 0
  for i, p in enumerate(p_ids_updated.tolist()):
    if mask[i]:
      save_dict[str(p)].append(x[i, :].detach().cpu())
      num_updated += 1
  assert num_updated == torch.sum(mask)
  return save_dict


def save_model_concepts(concepts, con_f):
  print("Saving to %s" % con_f)
  
  # Save as torch object
  with open(con_f + ".pth", 'wb') as f:
    torch.save(concepts, f)

  # Save as hdf5
  save_hdf5({"test": concepts}, con_f + ".hdf5")


def save_model_preds(model_preds, out_f):
  print("Saving to %s" % out_f)

  # Save as torch object
  with open(out_f + ".pth", 'wb') as f:
    torch.save(model_preds, f)

  # Save as hdf5
  save_hdf5({"test": model_preds}, out_f + ".hdf5")


def save_hdf5(data_dict, output_f):
  """
  data_dict = {group1: {
                        dataset1: np.array(),
                        dataset2: np.array(),
                       },
               group2: {
                        dataset1: np.array(),
                        dataset2: np.array(),
                       },
              }
  """
  f = h5py.File(output_f, "w")
  for group_name in data_dict:
    group = f.create_group(group_name)
    for dataset_name in data_dict[group_name]:
      dataset = group.create_dataset(dataset_name, data = data_dict[group_name][dataset_name])
  f.close()


def load_hdf5(input_f, as_dict = False):
  f = h5py.File(input_f, "r")
  data_dict = dict()
  if as_dict:
    for group_name in f:
      data_dict[group_name] = dict()
      for dataset_name in f[group_name]:
        data_dict[group_name][dataset_name] = f[group_name][dataset_name][:]
  return f, data_dict


def loop_batch(stage, batch_sz, split, loader, model, train_loader, action_embs, temporal_len, loss_func, con_loss_func = None, optimizer = None, save_concepts = False, save_preds = False, time_skip = False, device = 'cpu', wandb = None):

  overall_loss = 0
  overall_recon_loss = 0
  all_metrics = {"r2": [], "rmse": [], "mae": [], "mape": [], "sim": [], "entropy": []}
  get_time_vectorize = np.vectorize(get_time)
  batch_pts = loader[0]["subject_id"].unique().tolist()

  all_model_preds = {str(k): [] for k in batch_pts}
  all_concepts = {str(k): [] for k in batch_pts}
  if stage != "test": random.shuffle(batch_pts)
  for iteration, p_idx in enumerate(torch.split(torch.tensor(batch_pts), batch_sz)):
    print("[%s] Running on batch %d..." % (stage, iteration))
    
    p_x, p_dates, p_actions, p_data = get_dense_subject_data(loader[0], p_idx, split, loader[1], action_embs, (stage != "test"), time_skip)
    p_actions = p_actions.to(device)
    if "concept_ids" not in all_concepts: all_concepts["concept_ids"] = list(p_data.columns) # Save concept IDs

    if stage == "train": optimizer.zero_grad()
    
    if split == "patient" or split == "state": time_offset = 1
    elif split == "time":
      if stage == "train": time_offset = 1
      else: time_offset = 0
    
    if split == "time": # Get patient data from train_loader
      train_patient_x, train_patient_dates, train_patient_actions, train_patient_data = get_dense_subject_data(train_loader[0], p_idx, split, train_loader[1], action_embs, (stage != "test"), time_skip)
      x_train, x_train_mask = get_patient_tensor(train_patient_x, device)
      dates_train = get_time_vectorize(train_patient_dates)

    # Forward pass
    model_preds = []
    batch_loss = 0
    batch_recon_loss = 0
    timepoints = p_x.shape[1]
    if split == "time" and stage != "train": temporal_len = 0 # Reduce offset for next predictions (historical data is already provided)
    for time_idx in range(temporal_len, timepoints - time_offset):
      
      # Prepare output ground truth
      dates = get_time_vectorize(p_dates)
      x_curr, x_curr_mask = get_patient_tensor(p_x, device)
      x_next = x_curr[:, time_idx + time_offset, :] # Predict next time point
      x_next_mask = x_curr_mask[:, time_idx + time_offset, :]
      next_date = convert_dates(dates[:, time_idx + time_offset], device)
      next_date_mask = get_dates_mask(dates[:, time_idx + time_offset])
      if split == "state": next_action = p_actions[:, time_idx, :] # Shift index only for cellular data: state_1 is achieved via action_0 on state_0 (this is due to the formatting of the dataset)
      else: next_action = p_actions[:, time_idx + time_offset, :] # For patients, state_1 is achieved via action_1 on state_0

      # Prepare input data
      has_mask = True
      if (split == "patient" or split == "state") or (split == "time" and stage == "train"):
        if time_skip: # Only use historical data
          x_curr = x_curr[:, :(temporal_len + 1), :]
          x_curr_mask = x_curr_mask[:, :(temporal_len + 1), :]
          x_curr_mask = agg_mask(x_curr_mask, dates[:, 0:temporal_len + 1], device)
          dates = dates[:, 0:temporal_len + 1]
        else: # Use all data until current time
          x_curr = x_curr[:, :(time_idx + 1), :]
          x_curr_mask = x_curr_mask[:, :(time_idx + 1), :]
          x_curr_mask = agg_mask(x_curr_mask, dates[:, 0:time_idx + 1], device)
          dates = dates[:, 0:time_idx + 1]
      elif split == "time" and stage != "train": # Get all time points' data from train
        x_curr = x_train.clone()
        x_curr_mask = agg_mask(x_train_mask, dates_train, device)
        dates = dates_train
      else:
        raise NotImplementedError
      p_ids_updated = p_idx
      assert dates.shape[0:1] == x_curr.shape[0:1], (dates.shape, x_curr.shape)
      dates = convert_dates(dates, device)

      # Catch all nans for a patient (ideally, data is processed to avoid this)
      all_nans_mask = (x_curr_mask.sum(-1) / x_curr_mask.shape[-1]) == 1
      if all_nans_mask.sum() != 0:
        x_curr_mask[all_nans_mask, :] = False
        assert ((x_curr_mask.sum(-1) / x_curr_mask.shape[-1]) == 1).sum() == 0

      assert x_curr.dim() == 3, x_curr.dim() # Batch x Visits x Labs
      assert x_next.dim() == 2, x_next.dim() # Batch x Labs
      assert x_next_mask.dim() == 2, x_next_mask.dim() # Batch x Labs
      assert dates["year"].dim() == 2, dates.dim() # Batch x Visits
      assert next_date["year"].dim() == 2, next_date["year"].dim() # Batch x Visits

      out = model(x_curr, x_curr_mask.to(device), dates, next_date, next_action, has_mask, device = device)
      if model.nconcepts != 0:
        x_next_hat, concept_vec = out

        entr_score = calc_entropy(concept_vec)
        all_metrics["entropy"].append(entr_score)

        if save_concepts:
          all_concepts = iterate_updated_pids(all_concepts, p_ids_updated, concept_vec, ~next_date_mask.flatten())

        if concept_vec.shape[0] < len(p_idx):
            concept_vec_padded = torch.zeros((len(p_idx), concept_vec.shape[1]))
            indices_to_replace = torch.tensor([torch.where(p_idx == i)[0].item() for i in p_ids_updated])
            assert indices_to_replace.shape[0] == concept_vec.shape[0]
            concept_vec_padded[indices_to_replace] = concept_vec
            concept_vec = concept_vec_padded
            assert concept_vec.shape[0] == len(p_idx)
        
      else:
        x_next_hat = out
        entr_score = 0

      assert x_next_hat.shape == x_next_mask.shape, (x_next_hat.shape, x_next_mask.shape)
      assert x_next_hat.shape == x_next.shape
      assert x_next.shape == x_next_mask.shape

      if save_preds:
        all_model_preds = iterate_updated_pids(all_model_preds, p_ids_updated, x_next_hat, ~next_date_mask.flatten())

      # Filter for patients with timepoint
      x_next_hat = x_next_hat[~next_date_mask.flatten(), :]
      x_next = x_next[~next_date_mask.flatten(), :]

      if x_next_hat.shape[0] > 0:

        # Filter for lab tests with actual measurements
        x_next_hat = x_next_hat[~x_next_mask[~next_date_mask.flatten()]]
        x_next = x_next[~x_next_mask[~next_date_mask.flatten()]]
        if len(x_next) == 0 or len(x_next_hat) == 0: continue

        recon_loss = loss_func(x_next_hat, x_next)
        loss = recon_loss
        batch_loss += loss.item()
        batch_recon_loss += recon_loss.item()

        # Calculate metrics
        r2, rmse, mae, mape, sim, all_metrics = calc_metrics(tensor_dcn(x_next), tensor_dcn(x_next_hat), all_metrics)

        if wandb is not None:
          batch_stats = {"Batch %s Loss" % stage.capitalize(): loss,
                         "Batch %s R2" % stage.capitalize(): r2,
                         "Batch %s RMSE" % stage.capitalize(): rmse,
                         "Batch %s MAE" % stage.capitalize(): mae,
                         "Batch %s SIM" % stage.capitalize(): sim,
                         "Batch %s Entropy" % stage.capitalize(): entr_score}
          wandb.log(batch_stats)

        # Backpropagation
        if stage == "train" and not model.concept_ones:
          loss.backward()
          optimizer.step()

    # Record overall loss
    if timepoints > 0:
      overall_loss += (batch_loss/timepoints)
      overall_recon_loss += (batch_recon_loss/timepoints)
    
  all_metrics = {k: np.nanmean(v) if len(v) > 0 else None for k, v in all_metrics.items()}
  return overall_loss, overall_recon_loss, all_concepts, all_metrics, model, all_model_preds


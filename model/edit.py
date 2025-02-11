# Make fine edits to patient trajectories

import wandb
import numpy as np
from datetime import timedelta, datetime
import torch
from torch.optim import Adam

from model_utils import parse_args, set_seed, get_hparams, load_split_data, load_checkpoint, get_time, convert_dates, get_action_emb, get_patient_tensor, tensor_dcn, save_model_preds
from model import CLEF


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda": print(torch.cuda.get_device_name(0))


def edit(edit_actions, edit_length, edit_loader, action_embs, model, temporal_len, con_att):
    data = edit_loader[0]
    get_time_vectorize = np.vectorize(get_time)
    increment_t = timedelta(days = 1) # Amount of time to add to the last date

    patient_preds = {"concepts": []}
    for pt_idx, pt in enumerate(data["subject_id"].unique().tolist()):
        if pt_idx % 100 == 0: print("Iteration", pt_idx, "Running on patient:", pt)

        # Format data and extract temporal_len visits
        p_data = data[data["subject_id"] == pt]
        p_data = p_data.pivot(index=["subject_id", "charttime", "action"], columns="itemid", values="valuenum")
        if p_data.shape[0] <= temporal_len + 1: continue
        p_data = p_data.iloc[:temporal_len + 1]
        if len(p_data.columns) % 2 == 1: p_data["pad"] = np.nan
        if len(p_data.columns) != model.nconcepts: continue
        max_time = p_data.shape[0]
        
        p_x = torch.tensor(p_data.to_numpy().reshape(1, max_time, p_data.shape[1])) # Batch x Visits x Labs
        p_x, p_mask = get_patient_tensor(p_x, device)
        p_dates = [i[1] for i in p_data.index.tolist()]
        
        next_date = datetime.fromisoformat(p_dates[-1])
        for t in range(edit_length):
            next_date = next_date + increment_t
            p_dates.append(str(next_date))
        p_dates = np.array(p_dates).reshape(1, len(p_dates))
        
        # Get edit vectors
        edit_vec = torch.tensor([edit_actions[c] if c in edit_actions else float('nan') for c in p_data.columns]).reshape(1, -1).to(device)
        if len(patient_preds["concepts"]) == 0: patient_preds["concepts"] = p_data.columns.tolist()
        else: assert patient_preds["concepts"] == p_data.columns.tolist()

        for t in range(temporal_len, p_dates.shape[1] - 1):

            curr_dates = convert_dates(get_time_vectorize(p_dates[:,:t + 1]), device)
            next_date = convert_dates(get_time_vectorize(p_dates[:,t + 1]), device)
            assert curr_dates["year"].shape[1] == p_x.shape[1], (curr_dates["year"].shape[1], p_x.shape[1])

            X_t, c_t = model.inference_edit(edit_vec, p_x, p_mask, curr_dates, next_date, action_embs["None"].to(device), True, device = device)
            
            # Concatenate prediction X_t to input
            p_x = torch.concat([p_x, X_t.unsqueeze(1)], dim = 1)
            p_x, p_mask = get_patient_tensor(p_x, device)

        # Save predictions
        patient_preds[str(pt)] = tensor_dcn(p_x)

    print("Number of edited patients:", len(patient_preds))
    return patient_preds


def main():
  
    # Parse args and hparams
    args = parse_args()
    hparams_raw = get_hparams(args)

    # Set seed
    set_seed(args.seed)

    # Load data
    print("Loading data...")
    data, action_embs, _, _, edit_loader, hparams_raw = load_split_data(args.data_dir, hparams_raw, args.data_type)
    print("Edit dataset:", len(edit_loader[0]))

    # Initialize model
    print("Initializing model...")
    print(hparams_raw)
    wandb.init(config = hparams_raw, project = "clef-icml", entity = "sc-drug")
    hparams = wandb.config
    model = CLEF(seq_encoder = hparams["seq_encoder"],
                 nfeat = hparams["nfeat"],
                 nconcepts = hparams["nconcepts"],
                 ntoken = hparams["ntoken"],
                 nhead = hparams["nhead"],
                 nlayers = hparams["nlayers"],
                 condition = hparams["condition"],
                 condition_sz = hparams["condition_sz"],
                 apply_ct_proj = hparams["apply_ct_proj"],
                 concept_ones = hparams["concept_ones"],
                 conv_sz = hparams["conv_sz"],
                 qkv_proj_sz = hparams["qkv_proj_sz"],
                 dropout = hparams["dropout"]
                )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr = hparams["lr"])
    print("Loading model checkpoint at %s" % args.resume)
    model, optimizer = load_checkpoint(args.resume, model, optimizer)
    best_model_dict = {"best_model": model.to(device)}
    wandb.watch(model)
    print(best_model_dict)

    print("Starting inference...")
    # MIMIC edits
    if args.save_prefix == "MIMIC_t1d_glucose_":
        edit_actions = {50931: 0.5}
    elif args.save_prefix == "MIMIC_t1d_glucose_opposite_":
        edit_actions = {50931: 2}
    elif args.save_prefix == "MIMIC_t1d_wbc_":
        edit_actions = {51301: 0.5}
    elif args.save_prefix == "MIMIC_t1d_glucose_wbc_":
        edit_actions = {50931: 0.5, 51301: 0.5}
    
    # eICU edits
    elif args.save_prefix == "eICU_t1d_glucose_":
        edit_actions = {"glucose": 0.5}
    elif args.save_prefix == "eICU_t1d_glucose_opposite_":
        edit_actions = {"glucose": 2}
    elif args.save_prefix == "eICU_t1d_wbc_":
        edit_actions = {"WBC x 1000": 0.5}
    elif args.save_prefix == "eICU_t1d_glucose_wbc_":
        edit_actions = {"glucose": 0.5, "WBC x 1000": 0.5}
    else:
        raise NotImplementedError
    print("Edit actions:", edit_actions)
    edit_length = 10
    edit_preds = edit(edit_actions, edit_length, edit_loader, action_embs, best_model_dict["best_model"], hparams["temporal_len"], hparams["con_att"])

    print("Saving predictions...")
    save_model_preds(edit_preds, args.output_dir + args.save_prefix + "edit_preds")

    print("Finished.")


if __name__ == "__main__":
    main()


import time
import math
import copy
import torch
from torch.optim import Adam

import wandb

from model_utils import parse_args, set_seed, get_hparams, load_split_data, check_do_train, load_checkpoint, loop_batch, get_loss_func, save_model_concepts, save_model_preds
from model import CLEF


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda": print(torch.cuda.get_device_name(0))


def train(epochs, batch_sz, split, train_loader, val_loader, action_embs, model, temporal_len, optimizer, loss, con_loss, save_model_f, save_concepts, save_preds = False, time_skip = False):

  metrics = {"train_loss": [], "val_loss": [],
             "train_r2": [], "train_rmse": [], "train_mae": [], "train_mape": [], "train_entropy": [],
             "val_r2": [], "val_rmse": [], "val_mae": [], "val_mape": [], "val_entropy": []}
  best_model_dict = {"best_model": None, "best_val": math.inf}
  model.train()
  for epoch in range(epochs):
    start = time.time() 

    # Train Loop
    overall_train_loss, overall_train_recon_loss, all_train_pt_concepts, train_metrics, model, _ = loop_batch("train", batch_sz, split, train_loader, model, train_loader, action_embs, temporal_len, loss, con_loss, optimizer, save_concepts, save_preds, time_skip, device, wandb)

    # Validation loop
    overall_val_loss, overall_val_recon_loss, all_val_pt_concepts, val_metrics, model, _ = loop_batch("val", batch_sz, split, val_loader, model, train_loader, action_embs, temporal_len, loss, con_loss, save_concepts = save_concepts, save_preds = save_preds, time_skip = time_skip, device = device, wandb = wandb)

    # Save model
    if val_metrics["rmse"] < best_model_dict["best_val"]: # Minimize RMSE
      with open(save_model_f + ("model_ckpt_split=%s_temporal=%d_epoch=%d_valrmse=%.2f.pth" % (split, temporal_len, epoch, val_metrics["rmse"])), 'wb') as f:
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, f)
      best_model_dict["best_model"] = copy.deepcopy(model)
      best_model_dict["best_val"] = val_metrics["rmse"]

    # Record metrics
    end = time.time()
    train_stats = {"Epoch": epoch + 1,
                   "Total Train Loss": overall_train_loss,
                   "Total Train Recon Loss": overall_train_recon_loss,
                   "Total Train R2": train_metrics["r2"],
                   "Total Train RMSE": train_metrics["rmse"],
                   "Total Train MAE": train_metrics["mae"],
                   "Total Train SIM": train_metrics["sim"],
                   "Total Train Entropy": train_metrics["entropy"],
                  }
    print(train_stats)
    val_stats = {"Total Val Loss": overall_val_loss,
                 "Total Val Recon Loss": overall_val_recon_loss,
                 "Total Val R2": val_metrics["r2"],
                 "Total Val RMSE": val_metrics["rmse"],
                 "Total Val MAE": val_metrics["mae"],
                 "Total Val SIM": val_metrics["sim"],
                 "Total Val Entropy": val_metrics["entropy"],
                 "Time Lapsed": round(end - start, 2)}
    print(val_stats)
    wandb.log({**train_stats, **val_stats})
    metrics["train_loss"].append(overall_train_loss)
    metrics["val_loss"].append(overall_val_loss)
    metrics["train_r2"].append(train_metrics["r2"])
    metrics["train_rmse"].append(train_metrics["rmse"])
    metrics["train_mae"].append(train_metrics["mae"])
    metrics["train_mape"].append(train_metrics["mape"])
    metrics["train_entropy"].append(train_metrics["entropy"])
    metrics["val_r2"].append(val_metrics["r2"])
    metrics["val_rmse"].append(val_metrics["rmse"])
    metrics["val_mae"].append(val_metrics["mae"])
    metrics["val_mape"].append(val_metrics["mape"])
    metrics["val_entropy"].append(val_metrics["entropy"])

  print("Finish training!\n")
  return metrics, all_train_pt_concepts, all_val_pt_concepts, best_model_dict


def test(batch_sz, split, train_loader, test_loader, action_embs, model, temporal_len, loss, con_loss, save_concepts, save_preds, time_skip):
  #device = torch.device("cpu")
  model.to(device)
  model.eval()
  start = time.time()
  overall_test_loss, overall_test_recon_loss, all_test_concepts, test_metrics, _, all_test_preds = loop_batch("test", batch_sz, split, test_loader, model, train_loader, action_embs, temporal_len, loss, con_loss, save_concepts = save_concepts, save_preds = save_preds, time_skip = time_skip, device = device, wandb = wandb)
  end = time.time()
  test_stats = {"Total Test Loss": overall_test_loss,
                "Total Test Recon Loss": overall_test_recon_loss,
                "Total Test R2": test_metrics["r2"],
                "Total Test RMSE": test_metrics["rmse"],
                "Total Test MAE": test_metrics["mae"],
                "Total Test SIM": test_metrics["sim"],
                "Total Test Entropy": test_metrics["entropy"],
                "Time Lapsed": round(end - start, 2)}
  print(test_stats)
  wandb.log(test_stats)
  return overall_test_loss, all_test_concepts, all_test_preds, test_metrics


def main():
  
  # Parse args and hparams
  args = parse_args()
  hparams_raw = get_hparams(args)

  # Set seed
  set_seed(args.seed)

  # Load data
  print("Loading data...")
  data, action_embs, train_loader, val_loader, test_loader, hparams_raw = load_split_data(args.data_dir, hparams_raw, args.data_type)
  print("Train:", len(train_loader[0]))
  print("Val:", len(val_loader[0]))
  print("Test:", len(test_loader[0]))

  # Initialize model
  print("Initializing model...")
  print(hparams_raw)
  wandb.init(config = hparams_raw, project = "clef-icml", entity = "sc-drug")
  hparams = wandb.config
  do_train = check_do_train(args, hparams)
  loss, con_loss = get_loss_func(hparams["loss"])
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
  if args.resume != "" and not hparams["concept_ones"]:
    print("Loading model checkpoint at %s" % args.resume)
    model, optimizer = load_checkpoint(args.resume, model, optimizer)
  best_model_dict = {"best_model": model.to(device)}
  wandb.watch(model)

  # Train
  if do_train:
    print("Starting training...")
    train_metrics, _, _, best_model_dict = train(hparams["epochs"], hparams["batch_sz"], hparams["split"],
                                                 train_loader, val_loader, action_embs, model,
                                                 hparams["temporal_len"],
                                                 optimizer, loss, con_loss,
                                                 args.output_dir + args.save_prefix, args.save_concepts, time_skip = args.time_skip)
  print(best_model_dict)

  if not hparams["counterfactual"]:

    # Test
    print("Starting inference...")
    test_loss, all_test_concepts, all_test_preds, test_metrics = test(hparams["batch_sz"], hparams["split"],
                                                                         train_loader, test_loader, action_embs,
                                                                         best_model_dict["best_model"], 
                                                                         hparams["temporal_len"],
                                                                         loss, con_loss,
                                                                         args.save_concepts, args.save_preds, args.time_skip)
    
    # Save patient concepts
    if args.save_concepts:
      print("Saving concepts...")
      save_model_concepts(all_test_concepts, args.output_dir + args.save_prefix + ("pt_concepts_split=%s_temporal=%d" % (hparams["split"], hparams["temporal_len"])))
    
    # Save predictions
    if args.save_preds:
      print("Saving predictions...")
      save_model_preds(all_test_preds, args.output_dir + args.save_prefix + ("model_preds_split=%s_temporal=%d" % (hparams["split"], hparams["temporal_len"])))

  else:
    
    # Counterfactual generation
    print("Starting counterfactual generation...")
    
    cf_data = test_loader[0]
    for cf_type in cf_data["cf_type"].unique():
      cf_type_data = cf_data[cf_data["cf_type"] == cf_type]
      if cf_type != "original": cf_type_data["action"] = cf_type_data["cf_action"]
      print(cf_type_data)
      cf_type_loader = [cf_type_data, None]
      cf_loss, cf_concepts, cf_preds, cf_metrics = test(hparams["batch_sz"], hparams["split"],
                                                           [], cf_type_loader, action_embs,
                                                           best_model_dict["best_model"], 
                                                           hparams["temporal_len"],
                                                           loss, con_loss,
                                                           args.save_concepts, True, args.time_skip)

      # Save counterfactual patient concepts
      if args.save_concepts:
        print("Saving counterfactual concepts...")
        save_model_concepts(cf_concepts, args.output_dir + args.save_prefix + ("pt_concepts_cf=%s" % cf_type))
      
      # Save counterfactual predictions
      print("Saving counterfactual predictions...")
      save_model_preds(cf_preds, args.output_dir + args.save_prefix + ("model_preds_cf=%s" % cf_type))
  
  print("Finished.")


if __name__ == "__main__":
    main()

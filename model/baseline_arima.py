# Run ARIMA baselines
# (Wherever possible, mimic the process in train.py)
# (This baseline only applies to the time split)

import argparse
import torch
import random
import numpy as np

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.arima.model import ARIMA

import sys
sys.path.insert(0, '../') # Add our own scripts
from project_config import PROJECT_DIR
from model_utils import set_seed, load_split_data, get_dense_subject_data, get_patient_tensor, tensor_dcn, calc_metrics


def parse_args():
  parser = argparse.ArgumentParser(description="Train baseline ARIMA models.")

  # Model
  parser.add_argument("--model_type", type=str, default="VAR", choices=["VAR"], help="Model type")

  # Input/Output
  parser.add_argument("--data_type", type=str, default=None, choices=["patient", "cell"], help="Split data")
  parser.add_argument("--data_dir", type=str, default=None, help="Directory to processed EHR data")
  parser.add_argument("--split", type=str, default=None, choices=["patient", "state"], help="Split data")
  parser.add_argument("--time_skip", default=False, action="store_true", help="Predict immediate next time step t+1 (i.e., no skip; False) or future t+T (i.e., skip; True)")
  parser.add_argument("--output_dir", type=str, default=("%sresults/" % PROJECT_DIR), help="Output directory")
  parser.add_argument("--save_prefix", type=str, default="", help="Prefix of all saved files")
  
  # Training/inference
  parser.add_argument("--seed", type=int, default=1, help="Seed")

  args = parser.parse_args()
  return args


# Device
device = torch.device("cuda")
print("Using device:", device)


def run_model(model_type, train, num_steps):
    
    if model_type == "VAR":
        model = VAR(train).fit()
        preds = model.forecast(train, num_steps)
        return preds
    
    else:
        raise NotImplementedError
    
    preds = model.forecast(num_steps)
    return preds


def loop_batch(model_type, split, test_loader, time_skip, temporal_len = 8, time_offset = 1):

    test_metrics = {"r2": [], "rmse": [], "mae": [], "mape": [], "sim": [], "entropy": []}
    batch_pts = test_loader[0]["subject_id"].unique().tolist()

    for iteration, p_idx in enumerate(torch.split(torch.tensor(batch_pts), 1)):
        print("Running on batch %d (out of %d)..." % (iteration, len(batch_pts)))

        # Get patient data from test loader
        p_x, p_dates, _, p_data = get_dense_subject_data(test_loader[0], p_idx, split, test_loader[1], dict(), False, time_skip)
        x_curr, x_curr_mask = get_patient_tensor(p_x, device)
        timepoints = p_x.shape[1]

        # Generate
        for time_idx in range(temporal_len, timepoints - time_offset):

            # Prepare output ground truth
            x_next = x_curr[:, time_idx + time_offset, :] # Predict next time point
            assert x_next.shape[0] == 1

            # Prepare input data
            if time_skip:
                x_train = x_curr[:, :(temporal_len + 1), :]
            else:
                x_train = x_curr[:, :(time_idx + 1), :]
                assert x_train.shape[1] == time_idx + 1

            # Reshape for the individual patient
            x_train = x_train.squeeze(0)
            assert x_next.shape[0] == 1

            try:
                
                # Fit model
                if time_skip:
                    preds = run_model(model_type, tensor_dcn(x_train), (time_idx - temporal_len) + 1)[-1, :]
                else:
                    preds = run_model(model_type, tensor_dcn(x_train), 1)

                # Calculate metrics
                r2, rmse, mae, mape, sim, test_metrics = calc_metrics(tensor_dcn(x_next.flatten()), preds.flatten(), test_metrics)
            
            except Exception as e:
                print(f"An error occured: {e}")
                print(p_idx, time_idx, x_train.shape, x_next.shape)

        if iteration > 0 and iteration % 100 == 0: print("RMSE:", np.nanmean(test_metrics["rmse"]), "MAE:", np.nanmean(test_metrics["mae"]))

    test_metrics = {"Total Test %s" % k.upper(): np.nanmean(v) if len(v) > 0 else None for k, v in test_metrics.items()}
    print(test_metrics)
    return test_metrics


def main():

    # Parse args and hparams
    args = parse_args()
    hparams = {"split": args.split, "counterfactual": False, "spectra": ""}

    # Set seed
    set_seed(args.seed)

    # Load data
    print("Loading data...")
    data, _, train_loader, val_loader, test_loader, hparams = load_split_data(args.data_dir, hparams, args.data_type)
    print("Train:", len(train_loader[0]))
    print("Val:", len(val_loader[0]))
    print("Test:", len(test_loader[0]))

    # Loop through batches of patients to fit model and predict
    test_metrics = loop_batch(args.model_type, args.split, test_loader, args.time_skip)
    
    
if __name__ == "__main__":
    main()

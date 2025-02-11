#!/bin/bash


##############################
# WOT
##############################

# Step 1: Process raw data
python -u cell_data_process.py \
       --data_dir data/raw/WOT_traj_all \
       --raw True \
       --save_prefix data/WaddingtonOT/WOT_traj_all/

# Step 2: Generate splits (state-centric random split)
python -u cell_data_process.py \
       --data_dir data/raw/WOT_traj_all \
       --save_prefix data/WaddingtonOT/WOT_traj_all/ \
       --split state


##############################
# WOT-CF
##############################

# Step 1: Process raw data
python -u cell_data_process.py \
       --data_dir data/raw/WOT_counterfactual \
       --raw True \
       --save_prefix data/WaddingtonOT/WOT_counterfactual/

# Step 2: Generate splits (original versus counterfactual random split)
python -u cell_data_process.py \
       --data_dir data/raw/WOT_counterfactual \
       --save_prefix data/WaddingtonOT/WOT_counterfactual/ \
       --split counterfactual
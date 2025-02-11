#!/bin/bash

##############################
# MIMIC-IV
##############################

# Step 1: Process raw data
python -u data_process.py \
       --data_dir data/raw/MIMIC_IV/ \
       --raw True \
       --save_prefix data/MIMIC-IV/

# Step 2: Generate splits (patient-centric random split)
python -u data_process.py \
       --data_dir data/raw/MIMIC_IV/ \
       --save_prefix data/MIMIC-IV/ \
       --split patient


##############################
# eICU
##############################

# Step 1: Process raw data
python -u data_process.py \
       --data_dir data/raw/eICU/ \
       --raw True \
       --save_prefix data/eICU/

# Step 2: Generate splits (patient-centric random split)
python -u data_process.py \
       --data_dir data/raw/eICU/ \
       --save_prefix data/eICU/ \
       --split patient

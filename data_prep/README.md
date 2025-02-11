# Processing sequence data

## Cellular reprogramming experiments

To process the datasets of cellular reprogramming experiments, please use the script `cell_data_process.py`. The script is split into two steps: (1) Processing the raw trajectories and (2) Splitting the trajectories into train, validation, and test sets.

Example commands are provided in `run_cell_process.sh`

The final outputs of the script are (i.e., directly use for model):
- `filtered_data.csv`: Processed cellular trajectories
- `filtered_data_split=state.json`: Data split dictionary
- `action_embs_dict.pth`: Pretrained (frozen) condition embeddings


## Patient routine lab tests

To generate the datasets of patient routine lab tests from raw electronic health records (EHR), please use the script `data_process.py`. The script is split into two steps: (1) Processing the raw EHR data, such as identifying the most common lab tests, and (2) Splitting the trajectories into train, validation, and test sets.

Example commands are provided in `run_lab_process.sh`

The final outputs of the script are (i.e., directly use for model):
- `filtered_lab_code_data.csv`: Processed patient trajectories
- `filtered_lab_data_split=patient.json`: Data split dictionary
- `action_embs_dict.pth`: Pretrained (frozen) condition embeddings

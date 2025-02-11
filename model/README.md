# Running CLEF models


## Training

```
python -u train.py \
       --data_type ${data_type} --data_dir ${data_dir} --split ${split} \
       --seq_encoder ${seq_encoder} \
       --batch_sz ${batch_sz} \
       --seed ${seed} --save_prefix "${data_type}_${seq_encoder}_seed=${seed}_" \
       --best
```

How to use each flag:
- Use the `--data_type` (str) flag to specify the type of data to run CLEF on. Options for `${data_type}` are `cell` and `patient`
- Use the `--split` (str) flag to specify the type of data split. Options for `${split}` are `state` (for cellular trajectories) and `patient` (for patient trajectories)
- Use the `--apply_ct_proj` (bool) flag to add a single FFN layer to the concept encoder
- Use the `--seq_encoder` (str) flag to specify the sequence encoder. Options for `${seq_encoder}` are `transformer`, `xlstm`, and `moment` (Note: `moment_utils.py` loads the pretrained model checkpoint, so there is no need to pre-download it)
- Use the `--best` flag (bool) to use the best hyperparameters (as defined in the paper)


## Inference

```
python -u train.py \
       --data_type ${data_type} --data_dir ${data_dir} --split ${split} \
       --seq_encoder ${seq_encoder} \
       --batch_sz ${batch_sz} \
       --seed ${seed} --save_prefix "${data_type}_${seq_encoder}_seed=${seed}_" \
       --best --save_preds \
       --resume ${ckpt} --inference
```

How to use each flag (in addition to the flags for training):
- Use the `--save_preds` (bool) flag to save the model predictions
- Use the `--resume` (str) flag to specify the model checkpoint with which to run inference
- Use the `--inference` (bool) flag to run inference
- Use the `--time_skip` (bool) flag to run delayed sequence editing


## Edit

```
python -u edit.py \
       --data_type ${data_type} --data_dir ${data_dir} --split ${split} \
       --seq_encoder ${seq_encoder} \
       --batch_sz ${batch_sz} \
       --seed ${seed} --save_prefix "${edit}_seed=${seed}_" \
       --best --save_preds \
       --resume ${ckpt} --edit
```
How to use each flag (in addition to the flags for training):
- Use the `--save_preds` (bool) flag to save the model predictions
- Use the `--resume` (str) flag to specify the model checkpoint with which to run inference
- Use the `--edit` (bool) flag to run editing


## Baselines

To run baseline models with the sequence encoder ONLY, simply add the flag `--nconcepts 0` (int) to the above commands.

To run the SimpleLinear ablation, simply add the flag `--concept_ones` (bool) to the above commands (no training is required).

To run the traditional time series VAR model, run the script `baseline_arima.py`
```
python -u baseline_arima.py \
       --data_type ${data_type} --data_dir ${data_dir} --split ${split} \
       --model_type VAR \
       --seed ${seed} --save_prefix "${data_type}_VAR_seed=${seed}_" \
```

How to use each flag:
- Use the `--data_type` (str) flag to specify the type of data to run VAR on. Options for `${data_type}` are `cell` and `patient`
- Use the `--split` (str) flag to specify the type of data split. Options for `${split}` are `state` (for cellular trajectories) and `patient` (for patient trajectories)
- Use the `--time_skip` (bool) flag to run delayed sequence editing

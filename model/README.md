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
- Use the `--data_type` flag to specify the type of data to run CLEF on. Options for `${data_type}` are: `cell` and `patient`
- Use the `--split` flag to specify the type of data split. Options for `${split}` are: `state` (for cellular trajectories) and `patient` (for patient trajectories)
- Use the `--seq_encoder` flag to specify the sequence encoder. Options for `${seq_encoder}` are: `transformer`, `xlstm`, and `moment`
- Use the `--best` flag (boolean) to use the best hyperparameters (as defined in the paper)


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
- Use the `--save_preds` flag (boolean) to save the model predictions
- Use the `--resume` flag to specify the model checkpoint with which to run inference
- Use the `--inference` flag (boolean) to run inference
- Use the `--time_skip` flag (boolean) to run delayed sequence editing


## Edit

```
python -u edit.py \
       --data_type ${data_type} --data_dir ${data_dir} --split ${split} \
       --seq_encoder ${seq_encoder} \
       --batch_sz ${batch_sz} \
       --seed ${seed} --save_prefix "${edit}_seed=${seed}_" \
       --edit --best --save_preds \
       --resume ${ckpt}
```
How to use each flag (in addition to the flags for training):
- Use the `--save_preds` flag (boolean) to save the model predictions
- Use the `--resume` flag to specify the model checkpoint with which to run inference
- Use the `--edit` flag (boolean) to run editing
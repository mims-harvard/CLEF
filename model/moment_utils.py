
# Set up environment (run this before running the script)
#   conda install anaconda::pip  
#   pip install numpy pandas matplotlib tqdm
#   pip install git+https://github.com/moment-timeseries-foundation-model/moment.git


import torch
from momentfm import MOMENTPipeline


def get_moment_model(model_type):
    if model_type == "forecasting":
        model_kwargs = {
                        'task_name': 'forecasting',
                        'forecast_horizon': 192,
                        'head_dropout': 0.1,
                        'weight_decay': 0,
                        'freeze_encoder': True, # Freeze the patch embedding layer
                        'freeze_embedder': True, # Freeze the transformer encoder
                        'freeze_head': False, # The linear forecasting head must be trained
                       }

    elif model_type == "embedding":
        model_kwargs = {'task_name': 'embedding'} # We are loading the model in `embedding` mode to learn representations

    model = MOMENTPipeline.from_pretrained( "AutonLab/MOMENT-1-large", model_kwargs=model_kwargs)
    model.init()
    return model


from datetime import datetime
import math
import os
from config import Configuration as hypm

def get_rand_id():
    dt = datetime.now()
    return f"{math.floor(dt.timestamp())}"[2:]


def save_pipline(pipe):
    save_dir = hypm.save_model_dir
    os.makedirs(os.path.join(save_dir, hypm.exp_id), exist_ok=True)
    pipe.save_pretrained(os.path.join(save_dir, hypm.exp_id))


def save_exp_info(exp_id):

    filepath = f'logs/log_{exp_id}.txt'
    with open(filepath, 'w') as file:
        file.write(f'\nHyperparameter info: {datetime.now()}' + "\n\n")

        for k, v in hypm.__dict__.items():
            if not k.startswith("__") and not callable(v):
                file.write(f"{k}: {v}\n")
        file.write('\n\n')
    
        # df.to_string(file, index=True)

def write_to_file(expID, msg, content = ""):
    filepath = f'logs/log_{expID}.txt'
    with open(filepath, 'a') as file:
        file.write(f'\n{msg}')
        file.write(f'{content}\n')
    

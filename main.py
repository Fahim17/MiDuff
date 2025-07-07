import os
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from config import Configuration as hypm
from Uni_dataset import University1652Dataset, University1652Dataset2
from torch.utils.data import DataLoader
import pandas as pd
from eval import inference_generation, matching_score, matching_score2
from helper_func import get_rand_id, save_exp_info, save_pipline, write_to_file
from model_call import ImgEmbedFuseModel, getClipVisionModel, getStableDiffusionModel
from train import train_model, train_model_2
from diffusers import StableDiffusionPipeline


dataset_dir = hypm.dataset_path
torch.cuda.set_device(hypm.cuda_set_device)



# -------------------Dataset Loader drone to satellite-----------------
train_dataset = University1652Dataset(root_dir=dataset_dir, mode='train', num_input=hypm.num_of_encoder_img)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

test_dataset = University1652Dataset(root_dir=dataset_dir, mode = 'test', num_input=hypm.num_of_encoder_img, num_of_loc=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# -------------------Dataset Loader Street to drone-----------------
# train_dataset = University1652Dataset2(root_dir=dataset_dir, mode='train', num_input=hypm.num_of_encoder_img)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

# test_dataset = University1652Dataset2(root_dir=dataset_dir, mode = 'test', num_input=hypm.num_of_encoder_img)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


hypm.exp_id = get_rand_id()

def main():
    save_exp_info(exp_id=hypm.exp_id)

    clip_model = getClipVisionModel()

    # pipe = getStableDiffusionModel()
    # imgFuser = ImgEmbedFuseModel(numOfImg=hypm.num_of_encoder_img).to(device=hypm.device)
    # pipe.imgFuser = imgFuser

    # pipe, losses = train_model(train_data=train_dataset, train_loader = train_loader, img_model=clip_model, pipe=pipe)
    # df_loss = pd.DataFrame({'Loss': losses})
    # df_loss.to_csv(f'losses/losses_{hypm.exp_id}.csv')


    # if(hypm.save_model):
    #     save_pipline(pipe = pipe)

    # inference_generation(exp_id=hypm.exp_id, test_loader=test_loader, img_encoder=clip_model, pipe=pipe)

    # -------------------------------------Re-Training---------------------------------------
    # pre_exp_id = '51415276'
    # save_exp_info(exp_id=hypm.exp_id)
    # write_to_file(expID=hypm.exp_id, msg=f'Re-Training: {pre_exp_id}', content='*')

    # clip_model = getClipVisionModel()

    # pipe = StableDiffusionPipeline.from_pretrained(f"/home/fahimul/Documents/Research/MIDuff/trained_pipeline/{pre_exp_id}")
    # # imgFuser = ImgEmbedFuseModel(numOfImg=hypm.num_of_encoder_img).to(device=hypm.device)
    # # pipe.imgFuser = imgFuser

    # pipe, losses = train_model(train_data=train_dataset, train_loader = train_loader, img_model=clip_model, pipe=pipe)
    # df_loss = pd.DataFrame({'Loss': losses})
    # df_loss.to_csv(f'losses/losses_{hypm.exp_id}.csv')


    # if(hypm.save_model):
    #     save_pipline(pipe = pipe)

    # ------------------------------inference_generation---------------------------------
    # exp_id = '51428607'
    # print(f"Exp: {hypm.exp_id} Sleeping for 5 hours...")
    # time.sleep(5*60*60)
    # print(f"Done sleeping. Generating image for experiment {exp_id}")

    # pipe_inf = StableDiffusionPipeline.from_pretrained(f"/home/fahimul/Documents/Research/MIDuff/trained_pipeline/{exp_id}")
   
    # inference_generation(exp_id=exp_id, test_loader=test_loader, img_encoder=clip_model, pipe=pipe_inf)

    # ------------------------------stage 2 training---------------------------------
    exp_id = '51434428'
    stage2model, epoch_loss= train_model_2(exp_id=exp_id)
    matching_score2(exp_id=exp_id, stage2model=stage2model)
    # ------------------------------matching score---------------------------------
    # exp_id = '50834234'

    # matching_score(img_encoder=clip_model, exp_id=exp_id)






    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()




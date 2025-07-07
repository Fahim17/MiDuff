from email.headerregistry import DateHeader
import os
from datetime import datetime
import torch
import torch.optim as optim
from Uni_dataset import University1652Dataset_v3
from config import Configuration as hypm
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from helper_func import save_pipline, write_to_file
from loss import InfoNCE
from model_call import ImgEmbedFuseModel, TrainingStageModel
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()


def train_model(train_data, train_loader, img_model, pipe):
    # setup_ddp()
    # pipe = pipe.to(torch.cuda.current_device())
    # pipe.unet = DDP(pipe.unet, device_ids=[torch.cuda.current_device()])
    # pipe.vae = DDP(pipe.vae, device_ids=[torch.cuda.current_device()])

    pipe = pipe.to(hypm.device)
    unet = pipe.unet
    vae = pipe.vae
    # unet = unet.to(hypm.device)
    # vae = vae.to(hypm.device)
    img_model = img_model.to(hypm.device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=hypm.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epoch_loss = []

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     sampler=train_sampler,
    #     batch_size=hypm.batch,
    #     num_workers=4,
    # )

    print(20*'-'+'Training Start'+20*'-')
    write_to_file(expID=hypm.exp_id, msg="Training starts")
    for i in tqdm(range(hypm.epochs)):
        # train_sampler.set_epoch(i)
        running_loss = []
        for input_stack, target_image, folder_id in train_loader:
            # Encode 10 images

            inputs = input_stack.chunk(input_stack.shape[1], dim=1)  # [B, 3, H, W] x 10
            target_image = target_image[0].to(hypm.device)
        
            # print(inputs[0].shape)
            # embeddings = [clip_model(pixel_values=img).last_hidden_state.mean(dim=1) for img in inputs]
            embeddings = []
            for img in inputs:
                img = img.to(hypm.device)
                img = torch.squeeze(img, dim=1)
                embed = img_model(pixel_values=img).last_hidden_state
                embeddings.append(embed)

            #------------------------------Fusing methods v1-----------------------------------
            fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
            #------------------------------Fusing methods v2-----------------------------------
            # fused_embedding = pipe.imgFuser(x=torch.stack(embeddings))



            # VAE encode target image
            target_latents = vae.encode(target_image).latent_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, hypm.noise_time_step, (target_latents.shape[0],)).long().to(hypm.device)

            noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)
            # UNet prediction

            # print(f'noise_latents: {noisy_latents.shape}')
            # print(f'timesteps: {timesteps}')
            # print(f"fused_embeding: {fused_embedding.shape}")

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=fused_embedding).sample

            # Loss and backward
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.cpu().detach().numpy())
        
        if(i>0 and hypm.save_model and np.mean(running_loss)<min(epoch_loss) ):
            save_pipline(pipe = pipe)
            write_to_file(expID=hypm.exp_id, msg=f'pipeline_saved_on_epoch:', content=f'{i+1}')

        scheduler.step()

        epoch_loss.append(np.mean(running_loss))
        # write_to_file(expID=hypm.exp_id, msg=f'Loss_on_epoch:{i+1}=>', content=np.mean(epoch_loss[-1]))
    
    # cleanup_ddp()

    # Create a new pipeline with trained components
    pipe.unet = unet
    pipe.vae = vae

    write_to_file(expID=hypm.exp_id, msg="Training Ends: ", content=f"{datetime.now()}")


    return pipe, epoch_loss



def train_model_2(exp_id):

    test_dataset_gen = University1652Dataset_v3(root_dir=hypm.dataset_path, mode = 'test', num_input=1, exp_id=exp_id)
    test_loader_gen = DataLoader(test_dataset_gen, batch_size=32, shuffle=False, num_workers=4)

    stage2model = TrainingStageModel(input_dim=768).to(hypm.device)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.5)
    criterion = InfoNCE(loss_function=loss_fn,
                            device=hypm.device,
                            )

    optimizer = torch.optim.Adam(stage2model.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_loss = []


    print(20*'-'+'Training Stage 2 Start'+20*'-')
    write_to_file(expID=exp_id, msg=10*'-'+'Training Stage 2 Start'+10*'-')
    
    trainable_params = sum(p.numel() for p in stage2model.parameters() if p.requires_grad)
    write_to_file(expID=exp_id, msg="Trainable Parameters: ", content=f'{trainable_params:,}\n')

    for i in tqdm(range(400*60)):
        running_loss = []
        
        for input_stack, target_image, sat_target_img, folder_id in test_loader_gen:
            
            # print(input_stack)
            # print(target_image)
            # print(sat_target_img)
            # print(folder_id)
            input_stack = input_stack.to(hypm.device)
            target_image = target_image.to(hypm.device)
            sat_target_img = sat_target_img.to(hypm.device)

            input_stack = torch.squeeze(input_stack, dim=1)

            xdgs_embed, xs_embed = stage2model(xd=input_stack, xgs=target_image, xs=sat_target_img)


            loss = criterion(xdgs_embed, xs_embed)

       
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.cpu().detach().numpy())
        
        # if(i>0 and hypm.save_model and np.mean(running_loss)<min(epoch_loss) ):
        #     save_pipline(pipe = pipe)
        #     write_to_file(expID=hypm.exp_id, msg=f'pipeline_saved_on_epoch:', content=f'{i+1}')

        # scheduler.step()

        epoch_loss.append(np.mean(running_loss))
        write_to_file(expID=exp_id, msg=f'Stage 2 Loss_on_epoch:{i+1}=>', content=np.mean(epoch_loss[-1]))
    
   


    write_to_file(expID=hypm.exp_id, msg="Stage2 Training Ends: ", content=f"{datetime.now()}")


    return stage2model, epoch_loss




import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from Uni_dataset import University1652Dataset
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
from PIL import Image
from glob import glob
from torchvision import transforms as T
from transformers import CLIPModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from Uni_dataset import University1652Dataset_test
from torch.utils.data import DataLoader
from helper_func import get_rand_id

dataset_dir = "/home/fahimul/Documents/Research/Dataset/University-1651"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)

dataset = University1652Dataset(root_dir=dataset_dir, mode='train', num_input=10)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

exp_id = get_rand_id()
print(exp_id)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
clip_model.eval()


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
unet = pipe.unet
vae = pipe.vae

unet = unet.to(device)
vae = vae.to(device)
clip_model = clip_model.to(device)


def modify_unet(unet):
    for block in unet.down_blocks + unet.up_blocks:
        for attn in block.attentions:
            attn.encoder_hidden_states_dim = 512  # Match CLIP embedding size
    return unet


optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
epochs = 10
for i in tqdm(range(epochs)):
    for input_stack, target_image, folder_id in train_loader:
        # Encode 10 images

        inputs = input_stack.chunk(10, dim=1)  # [B, 3, H, W] x 10
        target_image = target_image[0].to(device)
    
        # print(inputs[0].shape)
        # embeddings = [clip_model(pixel_values=img).last_hidden_state.mean(dim=1) for img in inputs]
        embeddings = []
        for img in inputs:
            img = img.to(device)
            img = torch.squeeze(img, dim=1)
            embed = clip_model(pixel_values=img).last_hidden_state
            embeddings.append(embed)


        fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
        # VAE encode target image
        target_latents = vae.encode(target_image).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, 1000, (target_latents.shape[0],)).long().to(device)

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


# Create a new pipeline with trained components
pipe.unet = unet
pipe.vae = vae
# pipe.image_encoder = clip_model  # Optional, if used in generation
pipe.exp_id = exp_id

# Save to directory
save_dir = "/home/fahimul/Documents/Research/MIDuff/trained_pipeline"
os.makedirs(os.path.join(save_dir, exp_id), exist_ok=True)
pipe.save_pretrained(os.path.join(save_dir, exp_id))

# optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
# epochs = 1
# for i in tqdm(range(epochs)):
#     for input_stack, target_image, folder_id in train_loader:
#         # Encode 10 images

#         inputs = input_stack.chunk(10, dim=1)  # [B, 3, H, W] x 10
#         target_image = target_image[0].to(device)
    

#         # embeddings = [clip_model(pixel_values=img).last_hidden_state.mean(dim=1) for img in inputs]
#         embeddings = []
#         for img in inputs:
#             img = img.to(device)
#             img = torch.squeeze(img)
#             embed = clip_model(pixel_values=img).last_hidden_state
#             embeddings.append(embed)


#         fused_embedding = torch.mean(torch.stack(embeddings), dim=0)


#         # VAE encode target image
#         target_latents = vae.encode(target_image).latent_dist.sample() * 0.18215

#         # Add noise
#         noise = torch.randn_like(target_latents)
#         timesteps = torch.randint(0, 1000, (target_latents.shape[0],)).long().to(device)

#         noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)
#         # UNet prediction
#         print(f'noise_latents: {noisy_latents.shape}')
#         print(f'timesteps: {timesteps}')
#         print(f"fused_embeding: {fused_embedding.shape}")
#         noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=fused_embedding).sample

#         # Loss and backward
#         loss = F.mse_loss(noise_pred, noise)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()


# --------------------------------Testing--------------------------------------

# ------------------ Configuration ------------------


# # Paths
# input_dir = "your_input_images"  # Directory with input_01.jpg ... input_10.jpg
# unet_path = "path_to_trained_unet"  # e.g., ./checkpoints/unet/
output_path = "output"

# Image generation settings
num_inference_steps = 50
height, width = 224, 224
# -------------------Dataset Loader-----------------
test_dataset_dir = "/home/fahimul/Documents/Research/Dataset/University-1651"

# test_dataset = University1652Dataset_test(root_dir=test_dataset_dir, num_input=10, num_of_loc = 5)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
test_dataset = University1652Dataset(root_dir=dataset_dir, mode='train', num_input=10, num_of_loc = 5)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# ------------------ Load Models ------------------
print("Loading models...")
# unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)
# vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.eval().to(device)

# ------------------ Preprocessing ------------------
# clip_transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])  # CLIP normalization
# ])

# def encode_and_fuse(images):
#     with torch.no_grad():
#         inputs = torch.stack([clip_transform(img) for img in images]).to(device)  # [10, 3, 224, 224]
#         features = [clip_model(pixel_values=img.unsqueeze(0)).last_hidden_state.mean(dim=1) for img in inputs]
#         fused = torch.mean(torch.stack(features), dim=0)  # [1, 512]
#     return fused

# ------------------ Sampling Function ------------------
@torch.no_grad()
def generate_image(fused_embedding, steps=50, height=512, width=512):
    batch_size = fused_embedding.shape[0]
    latent = torch.randn((batch_size, 4, height // 8, width // 8)).to(device)
    scheduler.set_timesteps(steps)

    for t in scheduler.timesteps:
        latent_input = scheduler.scale_model_input(latent, t)

        # print(f'latent_input: {latent_input.shape}')
        # print(f'timesteps: {t}')
        # print(f"fused_embedding: {fused_embedding.shape}")

        noise_pred = unet(latent_input, t, encoder_hidden_states=fused_embedding).sample
        latent = scheduler.step(noise_pred, t, latent).prev_sample

    decoded = vae.decode(latent / 0.18215).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded[0]
    return T.ToPILImage()(decoded.squeeze().cpu())

# ------------------ Inference Entry Point ------------------

for input_stack, target_image, folder_id in test_loader:
    itm = 0
    embeddings = []

    inputs = input_stack.chunk(10, dim=1)  # [B, 3, H, W] x 10
    target_image = target_image[0].to(device)


    # embeddings = [clip_model(pixel_values=img).last_hidden_state.mean(dim=1) for img in inputs]
    
    print(f"Encoding and fusing input images of ID: {folder_id}")
    for img in tqdm(inputs):
        img = img.to(device)
        img = torch.squeeze(img, dim=1)
        embed = clip_model(pixel_values=img).last_hidden_state
        embeddings.append(embed)

    fused_cond = torch.mean(torch.stack(embeddings), dim=0)
    
    
    print(f"Generating image of ID: {folder_id}")
    result = generate_image(fused_cond, steps=num_inference_steps, height=height, width=width)

    # fused_cond = encode_and_fuse(img_x)
    
    os.makedirs(os.path.join(output_path, exp_id), exist_ok=True)

    # os.makedirs(os.path.dirname(f'{output_path}/{folder_id}.png'), exist_ok=True)
    result.save(f'{output_path}/{exp_id}/{folder_id[0]}.png')
    print(f"Saved output of: {folder_id}")


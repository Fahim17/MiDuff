from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
import torch
import torch.nn as nn

# Load pre-trained components
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

image_encoder = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.AdaptiveAvgPool2d((16, 16)),
)  # A lightweight encoder

# Custom fusion module
class MultiImageFusion(nn.Module):
    def __init__(self, input_dim=128, num_inputs=10):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):  # inputs: [B, 10, C, H, W]
        B, N, C, H, W = inputs.shape
        x = inputs.view(B * N, C, H, W)
        x = image_encoder(x)           # [B*N, 128, 16, 16]
        x = x.view(B, N, 128, -1)      # [B, 10, 128, 256]
        x = x.mean(-1)                 # [B, 10, 128]
        attn_out, _ = self.attn(x, x, x)  # [B, 10, 128]
        fused = attn_out.mean(1)          # [B, 128]
        return fused.unsqueeze(1)         # [B, 1, 128]

fusion = MultiImageFusion()

# Training loop
def train_step(input_images, ground_truth):
    # input_images: [B, 10, 3, H, W]
    # ground_truth: [B, 3, H, W]
    
    # 1. Encode GT into latent space
    with torch.no_grad():
        latents = vae.encode(ground_truth).latent_dist.sample() * 0.18215

    # 2. Fuse image encodings
    cond = fusion(input_images)  # [B, 1, 128]

    # 3. Sample noise and add to latent
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),)).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # 4. Predict noise
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample

    # 5. Loss
    loss = nn.MSELoss()(noise_pred, noise)
    loss.backward()
    optimizer.step()

    return loss.item()

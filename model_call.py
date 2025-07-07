import torch
import torch.nn as nn
from transformers import CLIPModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from config import Configuration as hypm

# HuggingFace CLIP
def getClipVisionModel():

    clip_model = CLIPModel.from_pretrained(hypm.img_encoder).vision_model
    clip_model.eval()

    return clip_model


def getStableDiffusionModel():

    pipe = StableDiffusionPipeline.from_pretrained(hypm.sd_pipeline)

    return pipe



class ImgEmbedFuseModel(nn.Module):
    def __init__(self, numOfImg=10, hidden_dim=5):
        super(ImgEmbedFuseModel, self).__init__()
        self.linear1 = nn.Linear(numOfImg, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: (10, 1, 50, 768)
        x = x.squeeze(1)  # shape: (10, 50, 768)

        # Rearrange to (50, 768, 10) to apply linear layers across batch dim
        x = x.permute(1, 2, 0)  # (50, 768, 10)

        # Apply linear layers across the last dimension (which was the original batch dim)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)  # shape: (50, 768, 1)

        x = x.squeeze(-1)  # shape: (50, 768)
        x = x.unsqueeze(0)  # shape: (1, 50, 768)

        return x
    


class TrainingStageModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=5):
        super(TrainingStageModel, self).__init__()
        self.img_model = CLIPModel.from_pretrained(hypm.img_encoder).vision_model

        for param in self.img_model.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 768)
        self.activation = nn.ReLU()

    def forward(self, xd, xgs, xs):

        xd_embed = self.img_model(pixel_values=xd).last_hidden_state
        xgs_embed = self.img_model(pixel_values=xgs).last_hidden_state
        xs_embed = self.img_model(pixel_values=xs).last_hidden_state

        xdgs_embed = xd_embed+xgs_embed

        xdgs_embed = self.activation(self.linear1(xdgs_embed))
        xdgs_embed = self.activation(self.linear2(xdgs_embed))
        xdgs_embed = self.linear3(xdgs_embed)

        xdgs_embed = xdgs_embed[:,0,:]
        xs_embed = xs_embed[:,0,:]


        return xdgs_embed, xs_embed

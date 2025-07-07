import torch

class Configuration:

    Message: str = "Drone to satellite, Fuse: Mean stack"

    # model
    img_encoder: str = "openai/clip-vit-base-patch32"
    sd_pipeline: str = "CompVis/stable-diffusion-v1-4"
    save_model: bool = True
    save_model_dir: str = "/home/fahimul/Documents/Research/MIDuff/trained_pipeline"


    # data
    dataset_path: str = "/home/fahimul/Documents/Research/Dataset/University-1651"
    num_of_encoder_img = 5 # Number of images for each location 
    batch = 1


    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_set_device = 0

    # experiment
    exp_id = -1
    
    # Image generation settings
    save_inference_dir: str = "/home/fahimul/Documents/Research/MIDuff/output"
    num_inference_steps = 50
    infer_height, infer_width = 224, 224

    # train
    epochs: int = 100
    lr = 1e-5
    noise_time_step = 1000


    # others
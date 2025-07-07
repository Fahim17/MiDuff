import torch
from diffusers import DDIMScheduler
from Uni_dataset import University1652Dataset_inf, University1652Dataset_v3
from config import Configuration as hypm
from tqdm import tqdm
import os
from torchvision import transforms as T
from torch.utils.data import DataLoader
import copy
import numpy as np
import time

from helper_func import write_to_file


# ------------------ Sampling Function ------------------
@torch.no_grad()
def generate_image(fused_embedding, pipe, steps=50, height=512, width=512):
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    batch_size = fused_embedding.shape[0]
    latent = torch.randn((batch_size, 4, height // 8, width // 8)).to(hypm.device)
    scheduler.set_timesteps(steps)

    for t in scheduler.timesteps:
        latent_input = scheduler.scale_model_input(latent, t)

        # print(f'latent_input: {latent_input.shape}')
        # print(f'timesteps: {t}')
        # print(f"fused_embedding: {fused_embedding.shape}")

        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=fused_embedding).sample
        latent = scheduler.step(noise_pred, t, latent).prev_sample

    decoded = pipe.vae.decode(latent / 0.18215).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded[0]
    return T.ToPILImage()(decoded.squeeze().cpu())


def inference_generation(exp_id, test_loader, img_encoder, pipe):
    img_encoder = img_encoder.to(hypm.device)
    pipe = pipe.to(hypm.device)
    write_to_file(exp_id, f"Creating inference for:{exp_id}", content = "=>")

    print(20*'-'+'Image Generation Start'+20*'-')
    for input_stack, target_image, folder_id in test_loader:
        itm = 0
        embeddings = []

        inputs = input_stack.chunk(10, dim=1)  # [B, 3, H, W] x 10
        target_image = target_image[0].to(hypm.device)


        # embeddings = [clip_model(pixel_values=img).last_hidden_state.mean(dim=1) for img in inputs]
        
        print(f"Encoding and fusing input images of ID: {folder_id}")
        for img in tqdm(inputs):
            img = img.to(hypm.device)
            img = torch.squeeze(img, dim=1)
            embed = img_encoder(pixel_values=img).last_hidden_state
            embeddings.append(embed)

        #------------------------------Fusing methods v1-----------------------------------
        fused_cond = torch.mean(torch.stack(embeddings), dim=0)
        #------------------------------Fusing methods v2-----------------------------------
        # fused_cond = pipe.imgFuser(x=torch.stack(embeddings))
        
        
        # print(f"Generating image of ID: {folder_id}")
        result = generate_image(fused_cond, pipe = pipe, steps=hypm.num_inference_steps, height=hypm.infer_height, width=hypm.infer_width)

        # fused_cond = encode_and_fuse(img_x)
        
        os.makedirs(os.path.join(hypm.save_inference_dir, hypm.exp_id), exist_ok=True)

        # os.makedirs(os.path.dirname(f'{output_path}/{folder_id}.png'), exist_ok=True)
        result.save(f'{hypm.save_inference_dir}/{hypm.exp_id}/{folder_id[0]}.png')
        # print(f"Saved output of: {folder_id}")




def matching_score(img_encoder, verbose=True, exp_id=None):
    img_encoder = img_encoder.to(hypm.device)

    test_dataset_gen = University1652Dataset_inf(root_dir=hypm.dataset_path, mode = 'test', num_input=hypm.num_of_encoder_img, exp_id=exp_id)
    test_dataset_og = University1652Dataset_inf(root_dir=hypm.dataset_path, mode = 'test', num_input=hypm.num_of_encoder_img)
    test_loader_gen = DataLoader(test_dataset_gen, batch_size=1, shuffle=False, num_workers=4)
    test_loader_og = DataLoader(test_dataset_og, batch_size=1, shuffle=False, num_workers=4)


    street_features_list = []
    drone_feature_list = []
    drone_og_feature_list = []
    sat_feature_list = []
    ids_list = []

    if verbose:
        bar = tqdm(test_loader_gen, total=len(test_loader_gen))
    else:
        bar = test_loader_gen

    with torch.no_grad():

        for street, drone, sat, idx in bar:
            street, drone, sat = street.to(hypm.device), drone.to(hypm.device), sat.to(hypm.device)
            ids_list.append(int(idx[0]))

            street_embed = img_encoder(pixel_values=street).pooler_output
            drone_embed = img_encoder(pixel_values=drone).pooler_output
            sat_embed = img_encoder(pixel_values=sat).pooler_output

            street_features_list.append(street_embed)
            drone_feature_list.append(drone_embed)
            sat_feature_list.append(sat_embed)

        street_features = torch.cat(street_features_list, dim=0) 
        drone_features = torch.cat(drone_feature_list, dim=0)
        sat_features = torch.cat(sat_feature_list, dim=0)
        ids_list = torch.tensor(ids_list)
        # print(ids_list)
        # ids_list = torch.cat(ids_list, dim=0).to(hypm.device)


    if verbose:
        bar_og = tqdm(test_loader_og, total=len(test_loader_og))
    else:
        bar_og = test_loader_og

    with torch.no_grad():

        for street, drone, sat, idx in bar_og:
            street, drone, sat = street.to(hypm.device), drone.to(hypm.device), sat.to(hypm.device)

            # street_embed = img_encoder(pixel_values=street).pooler_output
            drone_og_embed = img_encoder(pixel_values=drone).pooler_output
            # sat_embed = img_encoder(pixel_values=sat).pooler_output

            # street_features_list.append(street_embed)
            drone_og_feature_list.append(drone_og_embed)
            # sat_feature_list.append(sat_embed)

        # street_features = torch.cat(street_features_list, dim=0) 
        drone_og_features = torch.cat(drone_og_feature_list, dim=0)
        # sat_features = torch.cat(sat_feature_list, dim=0)


    result = calculate_scores2(query_features=drone_features, reference_features=drone_og_features, query_labels=ids_list, reference_labels=ids_list, step_size=1000, ranks=[1, 5, 10])
    print(result)
    result =  accuracy(query_features=drone_features, reference_features=drone_og_features, query_labels=ids_list, topk=[1, 5, 10])
    print(result)
    result = compute_topk(query_embeddings=drone_features, gallery_embeddings=drone_og_features)
    print(result)


def matching_score2(exp_id=None, stage2model=None, verbose=True):

    test_dataset_gen = University1652Dataset_v3(root_dir=hypm.dataset_path, mode = 'test', num_input=1, exp_id=exp_id)
    test_loader_gen = DataLoader(test_dataset_gen, batch_size=32, shuffle=False, num_workers=4)


    xdgs_features_list = []
    xs_feature_list = []

    ids_list = []

    if verbose:
        bar = tqdm(test_loader_gen, total=len(test_loader_gen))
    else:
        bar = test_loader_gen

    with torch.no_grad():

        for dr, gen_sat, sat, idx in bar:
            dr, gen_sat, sat = dr.to(hypm.device), gen_sat.to(hypm.device), sat.to(hypm.device)
            dr = torch.squeeze(dr, dim=1)
            ids_list.append(int(idx[0]))

            xdgs_embed, xs_embed = stage2model(xd=dr, xgs=gen_sat, xs=sat)

            xdgs_features_list.append(xdgs_embed)
            xs_feature_list.append(xs_embed)

        xdgs_features = torch.cat(xdgs_features_list, dim=0) 
        xs_features = torch.cat(xs_feature_list, dim=0)
        ids_list = torch.tensor(ids_list)
        print(ids_list)


    # result = calculate_scores2(query_features=xdgs_features, reference_features=xs_features, query_labels=ids_list, reference_labels=ids_list, step_size=1000, ranks=[1, 5, 10])
    # print(result)
    # write_to_file(expID=exp_id, msg=f'SampleGeo: ', content=result)
    result =  accuracy(query_features=xdgs_features, reference_features=xs_features, query_labels=ids_list, topk=[1, 5, 10])
    print(result)
    write_to_file(expID=exp_id, msg=f'Mine: ', content=result)
    result = compute_topk(query_embeddings=xdgs_features, gallery_embeddings=xs_features)
    print(result)
    write_to_file(expID=exp_id, msg=f'ChatGPT: ', content=result)





    

def compute_topk(query_embeddings, gallery_embeddings, topk=(1, 5, 10)):
    query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
    gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, dim=1)
                                                   
    similarity = torch.matmul(query_embeddings, gallery_embeddings.T)
    correct = torch.arange(similarity.size(0)).to(hypm.device)  # [0, 1, ..., 700]
    _, indices = similarity.topk(max(topk), dim=1, largest=True)
    
    
    topk_results = {}
    for k in topk:
        match = (indices[:, :k] == correct.unsqueeze(1)).any(dim=1)
        topk_results[f"Top-{k}"] = match.float().mean().item()
    
    return topk_results





def calculate_scores2(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results



def accuracy(query_features, reference_features, query_labels, topk=[1,5,10], tv_all_reference_features = None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(f'query labels {query_labels}')
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(N//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    query_features = query_features.cpu()
    reference_features = reference_features.cpu()
    query_labels = query_labels.cpu()

    if N < 80000:
        query_features_norm = np.sqrt(np.sum((query_features**2).numpy(), axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum((reference_features ** 2).numpy(), axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).T)
        similarity = similarity.numpy()
        # print(similarity.shape)
        # save_tensor(var_name='similarity', var=similarity)
        #----------------------------------------------------------------------
        for i in range(N):
            # ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
            ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)
            # print(ranking)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
                # print(f'k: {k} == results: {results}')
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                   (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results

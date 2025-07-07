import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoProcessor
from config import Configuration as hypm
from pathlib import Path

# ------------------------------------Drone to satellite--------------------------
class University1652Dataset(Dataset):
    def __init__(self, root_dir, mode='train', num_input=5, num_of_loc = None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'University-1652/')
            mode (str): 'train' or 'test' (we assume paired folders exist under train)
            transform: PyTorch transforms
        """
        if(mode=='train'):
            self.drone_dir = os.path.join(root_dir, mode, 'drone')
            self.satellite_dir = os.path.join(root_dir, mode, 'satellite')
        elif(mode=='test'):
            self.drone_dir = os.path.join(root_dir, mode, 'query_drone')
            self.satellite_dir = os.path.join(root_dir, mode, 'query_satellite')
        else:
            raise Exception("Dataset not found!!")
        
        self.processor = AutoProcessor.from_pretrained(hypm.img_encoder)


        self.pairs = []  # (drone_path, satellite_path, folder_id)

        folders = sorted(os.listdir(self.drone_dir))

        if num_of_loc is not None:
            folders = folders[:num_of_loc]

        for _, folder in enumerate(folders):

            drone_folder = os.path.join(self.drone_dir, folder)
            satellite_folder = os.path.join(self.satellite_dir, folder)

            if not os.path.isdir(drone_folder) or not os.path.isdir(satellite_folder):
                continue

            # drone_imgs = [os.path.join(drone_folder, f) for f in os.listdir(drone_folder)
            #               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # satellite_imgs = [os.path.join(satellite_folder, f) for f in os.listdir(satellite_folder)
            #                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # for d_path in drone_imgs:
            #     for s_path in satellite_imgs:
            #         self.pairs.append((d_path, s_path, folder))

            drone_imgs = [os.path.join(drone_folder, f"image-{self.zero_padd(i+1)}.jpeg") for i in range(num_input)]
            satellite_imgs = os.path.join(satellite_folder, f"{folder}.jpg")
            self.pairs.append((drone_imgs, satellite_imgs, folder))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drone_paths, satellite_path, folder_id = self.pairs[idx]

        all_drone_image = []
        for img in drone_paths:
            drone_img = Image.open(img).convert('RGB')
            processed_img = self.processor(images=drone_img, return_tensors="pt")
            all_drone_image.append(processed_img.pixel_values)
        
        stacked_input = torch.cat(all_drone_image, dim=0)

        sat_img = Image.open(satellite_path).convert('RGB')
        target = self.processor(images = sat_img)
        target = target.pixel_values
        

        # drone_img = Image.open(drone_paths).convert('RGB')
        # satellite_img = Image.open(satellite_path).convert('RGB')

        # if self.transform:
        #     drone_img = self.transform(drone_img)
        #     satellite_img = self.transform(satellite_img)

        return stacked_input, target, folder_id
    
    def zero_padd(self, number):
        """
        Takes an integer and returns a string with a leading zero 
        if it is a single digit (0 to 9). Otherwise, returns the number as a string.
        """
        if 0 <= number < 10:
            return f"0{number}"
        else:
            return str(number)




# ------------------------------------Ground to Drone--------------------------


class University1652Dataset2(Dataset):
    def __init__(self, root_dir, mode='train', num_input=5, num_of_loc = None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'University-1652/')
            mode (str): 'train' or 'test' (we assume paired folders exist under train)
            transform: PyTorch transforms
        """
        if(mode=='train'):
            self.street_dir = os.path.join(root_dir, mode, 'street')
            self.drone_dir = os.path.join(root_dir, mode, 'drone')
        elif(mode=='test'):
            self.street_dir = os.path.join(root_dir, mode, 'query_street')
            self.drone_dir = os.path.join(root_dir, mode, 'query_drone')
        else:
            raise Exception("Dataset not found!!")
        
        self.processor = AutoProcessor.from_pretrained(hypm.img_encoder)


        self.pairs = []  # (drone_path, satellite_path, folder_id)

        folders = sorted(os.listdir(self.street_dir))

        if num_of_loc is not None:
            folders = folders[:num_of_loc]

        for _, folder in enumerate(folders):

            street_folder = os.path.join(self.street_dir, folder)
            drone_folder = os.path.join(self.drone_dir, folder)

            if not os.path.isdir(street_folder) or not os.path.isdir(drone_folder):
                continue
            

            str_dir = Path(street_folder)
            street_imgs = [os.path.join(street_folder,f.name) for f in str_dir.iterdir() if f.is_file()]
            drone_imgs = [os.path.join(drone_folder, f"image-{self.zero_padd(i+1)}.jpeg") for i in range(10)]
            # drone_imgs = os.path.join(drone_folder, f"{folder}.jpg")
            self.pairs.append((street_imgs, drone_imgs, folder))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        street_paths, drone_paths, folder_id = self.pairs[idx]

        all_str_image = []
        for img in street_paths:
            str_img = Image.open(img).convert('RGB')
            processed_img = self.processor(images=str_img, return_tensors="pt")
            all_str_image.append(processed_img.pixel_values)
        
        stacked_input = torch.cat(all_str_image, dim=0)

        drone_img = Image.open(random.choice(drone_paths)).convert('RGB')
        target = self.processor(images = drone_img)
        target = target.pixel_values
        

        # drone_img = Image.open(drone_paths).convert('RGB')
        # satellite_img = Image.open(satellite_path).convert('RGB')

        # if self.transform:
        #     drone_img = self.transform(drone_img)
        #     satellite_img = self.transform(satellite_img)

        return stacked_input, target, folder_id
    
    def zero_padd(self, number):
        """
        Takes an integer and returns a string with a leading zero 
        if it is a single digit (0 to 9). Otherwise, returns the number as a string.
        """
        if 0 <= number < 10:
            return f"0{number}"
        else:
            return str(number)



# ---------------------------------Ground, Drone, satellite--------------------------


class University1652Dataset_inf(Dataset):
    def __init__(self, root_dir, mode='test', num_input=5, num_of_loc = None, exp_id = None): # mode=og/gen
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'University-1652/')
            mode (str): 'train' or 'test' (we assume paired folders exist under train)
            transform: PyTorch transforms
        """
        self.exp_id = exp_id

        if(exp_id is not None):
            self.street_dir = os.path.join(root_dir, mode, 'query_street')
            self.drone_dir = os.path.join('/home/fahimul/Documents/Research/MIDuff/output', f'{exp_id}')
            self.sat_dir = os.path.join(root_dir, mode, 'query_satellite')
        elif(exp_id is None):
            self.street_dir = os.path.join(root_dir, mode, 'query_street')
            self.drone_dir = os.path.join(root_dir, mode, 'query_drone')
            self.sat_dir = os.path.join(root_dir, mode, 'query_satellite')

        else:
            raise Exception("Dataset not found!!")
        
        self.processor = AutoProcessor.from_pretrained(hypm.img_encoder)


        self.pairs = []  # (drone_path, satellite_path, folder_id)

        folders = sorted(os.listdir(self.street_dir))

        if num_of_loc is not None:
            folders = folders[:num_of_loc]

        for _, folder in enumerate(folders):

            street_folder = os.path.join(self.street_dir, folder)

            if exp_id is None:
                drone_folder = os.path.join(self.drone_dir, folder)
            else:
                drone_folder = os.path.join(self.drone_dir, f'{folder}.png')


            sat_folder = os.path.join(self.sat_dir, folder)


            if not os.path.isdir(street_folder) or not os.path.isdir(sat_folder):
                continue
            

            str_dir = Path(street_folder)
            street_imgs = [os.path.join(street_folder,f.name) for f in str_dir.iterdir() if f.is_file()]
            if exp_id is None:
                drone_imgs = [os.path.join(drone_folder, f"image-{self.zero_padd(i+1)}.jpeg") for i in range(10)]
            else:
                drone_imgs = drone_folder
            satellite_imgs = os.path.join(sat_folder, f"{folder}.jpg")
            self.pairs.append((street_imgs, drone_imgs, satellite_imgs, folder))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        street_paths, drone_paths, sat_paths, folder_id = self.pairs[idx]
        all_str_image = []
        for img in street_paths:
            str_img = Image.open(img).convert('RGB')
            processed_img = self.processor(images=str_img, return_tensors="pt")
            all_str_image.append(processed_img.pixel_values)
        
        stacked_input = torch.cat(all_str_image, dim=0)

        if self.exp_id is None:
            drone_img = Image.open(random.choice(drone_paths)).convert('RGB')
            target = self.processor(images = drone_img)
            target = target.pixel_values
        else:
            drone_img = Image.open(drone_paths).convert('RGB')
            target = self.processor(images = drone_img)
            target = target.pixel_values

        sat_img = Image.open(sat_paths).convert('RGB')
        sat_target = self.processor(images = sat_img)
        sat_target = sat_target.pixel_values

        

        # drone_img = Image.open(drone_paths).convert('RGB')
        # satellite_img = Image.open(satellite_path).convert('RGB')

        # if self.transform:
        #     drone_img = self.transform(drone_img)
        #     satellite_img = self.transform(satellite_img)

        return stacked_input[0], target[0], sat_target[0], folder_id
 

    
    def zero_padd(self, number):
        """
        Takes an integer and returns a string with a leading zero 
        if it is a single digit (0 to 9). Otherwise, returns the number as a string.
        """
        if 0 <= number < 10:
            return f"0{number}"
        else:
            return str(number)



# --------------------------------- drone, gen-satellite, satellite--------------------------

class University1652Dataset_v3(Dataset):
    def __init__(self, root_dir, mode='test', num_input=5, num_of_loc = None, exp_id = None): # mode=og/gen
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'University-1652/')
            mode (str): 'train' or 'test' (we assume paired folders exist under train)
            transform: PyTorch transforms
        """
        self.exp_id = exp_id

        if(exp_id is not None):
            self.img1_dir = os.path.join(root_dir, mode, 'query_drone')
            self.img2_dir = os.path.join('/home/fahimul/Documents/Research/MIDuff/output', f'{exp_id}')
            self.img3_dir = os.path.join(root_dir, mode, 'query_satellite')
        elif(exp_id is None):
            self.img1_dir = os.path.join(root_dir, mode, 'query_street')
            self.img2_dir = os.path.join(root_dir, mode, 'query_drone')
            self.img3_dir = os.path.join(root_dir, mode, 'query_satellite')

        else:
            raise Exception("Dataset not found!!")
        
        self.processor = AutoProcessor.from_pretrained(hypm.img_encoder)


        self.pairs = []  # (drone_path, satellite_path, folder_id)

        folders = sorted(os.listdir(self.img1_dir))

        if num_of_loc is not None:
            folders = folders[:num_of_loc]

        for _, folder in enumerate(folders):

            img1_folder = os.path.join(self.img1_dir, folder)

            if exp_id is None:
                img2_folder = os.path.join(self.img2_dir, folder)
            else:
                img2_folder = os.path.join(self.img2_dir, f'{folder}.png')


            img3_folder = os.path.join(self.img3_dir, folder)


            if not os.path.isdir(img1_folder) or not os.path.isdir(img3_folder):
                continue
            

            img1_dir = Path(img1_folder)
            img1_imgs = [os.path.join(img1_dir, f"image-{self.zero_padd(i+1)}.jpeg") for i in range(num_input)]
            img2_imgs = img2_folder

            img3_imgs = os.path.join(img3_folder, f"{folder}.jpg")
            self.pairs.append((img1_imgs, img2_imgs, img3_imgs, folder))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drone, gen_sat, sat, folder_id = self.pairs[idx]
        all_str_image = []
        for img in drone:
            drone_img = Image.open(img).convert('RGB')
            processed_img = self.processor(images=drone_img, return_tensors="pt")
            all_str_image.append(processed_img.pixel_values)
        
        stacked_input = torch.cat(all_str_image, dim=0)

        if self.exp_id is None:
            gen_sat_img = Image.open(random.choice(gen_sat)).convert('RGB')
            target = self.processor(images = gen_sat_img)
            target = target.pixel_values
        else:
            gen_sat_img = Image.open(gen_sat).convert('RGB')
            target = self.processor(images = gen_sat_img)
            target = target.pixel_values

        sat_img = Image.open(sat).convert('RGB')
        sat_target = self.processor(images = sat_img)
        sat_target = sat_target.pixel_values

        

        # drone_img = Image.open(drone_paths).convert('RGB')
        # satellite_img = Image.open(satellite_path).convert('RGB')

        # if self.transform:
        #     drone_img = self.transform(drone_img)
        #     satellite_img = self.transform(satellite_img)

        return stacked_input, target[0], sat_target[0], folder_id[0]
 

    
    def zero_padd(self, number):
        """
        Takes an integer and returns a string with a leading zero 
        if it is a single digit (0 to 9). Otherwise, returns the number as a string.
        """
        if 0 <= number < 10:
            return f"0{number}"
        else:
            return str(number)









class University1652Dataset_test(Dataset):
    def __init__(self, root_dir, mode='test', num_input=5, num_of_test_img = 5):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'University-1652/')
            mode (str): 'train' or 'test' (we assume paired folders exist under train)
            transform: PyTorch transforms
        """
            
        self.drone_dir = os.path.join(root_dir, mode, 'query_drone')
        self.satellite_dir = os.path.join(root_dir, mode, 'query_satellite')
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
      

        self.pairs = []  # (drone_path, satellite_path, folder_id)

        folders = sorted(os.listdir(self.drone_dir))
        
        folders = folders[:num_of_test_img]

        for _, folder in enumerate(folders):

            drone_folder = os.path.join(self.drone_dir, folder)
            satellite_folder = os.path.join(self.satellite_dir, folder)

            if not os.path.isdir(drone_folder) or not os.path.isdir(satellite_folder):
                continue

            # drone_imgs = [os.path.join(drone_folder, f) for f in os.listdir(drone_folder)
            #               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # satellite_imgs = [os.path.join(satellite_folder, f) for f in os.listdir(satellite_folder)
            #                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # for d_path in drone_imgs:
            #     for s_path in satellite_imgs:
            #         self.pairs.append((d_path, s_path, folder))

            drone_imgs = [os.path.join(drone_folder, f"image-{self.zero_padd(i+1)}.jpeg") for i in range(num_input)]
            satellite_imgs = os.path.join(satellite_folder, f"{folder}.jpg")
            self.pairs.append((drone_imgs, satellite_imgs, folder))

        


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drone_paths, satellite_path, folder_id = self.pairs[idx]

        all_drone_image = []
        for img in drone_paths:
            drone_img = Image.open(img).convert('RGB')
            processed_img = self.processor(images=drone_img, return_tensors="pt")
            all_drone_image.append(processed_img.pixel_values)
        
        stacked_input = torch.cat(all_drone_image, dim=0)

        sat_img = Image.open(satellite_path).convert('RGB')
        target = self.processor(images = sat_img)
        target = target.pixel_values
        

        # drone_img = Image.open(drone_paths).convert('RGB')
        # satellite_img = Image.open(satellite_path).convert('RGB')

        # if self.transform:
        #     drone_img = self.transform(drone_img)
        #     satellite_img = self.transform(satellite_img)

        return stacked_input, target, folder_id
    
    def zero_padd(self, number):
        """
        Takes an integer and returns a string with a leading zero 
        if it is a single digit (0–9). Otherwise, returns the number as a string.
        """
        if 0 <= number < 10:
            return f"0{number}"
        else:
            return str(number)





# import os
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader


# class University1652Dataset(Dataset):
#     def __init__(self, root_dir, mode='train', view='drone', transform=None):
#         """
#         Args:
#             root_dir (str): Base directory of dataset (e.g., 'University-1652/')
#             mode (str): 'train' or 'test'
#             view (str): 'drone', 'satellite', or 'query'/'gallery' if mode is 'test'
#             transform: PyTorch transforms to apply
#         """
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []

#         view_dir = os.path.join(root_dir, mode, view)
#         if not os.path.exists(view_dir):
#             raise ValueError(f"Directory not found: {view_dir}")

#         pid = 0  # person ID / place ID
#         for folder in sorted(os.listdir(view_dir)):
#             folder_path = os.path.join(view_dir, folder)
#             if not os.path.isdir(folder_path):
#                 continue
#             for img_file in os.listdir(folder_path):
#                 if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     self.image_paths.append(os.path.join(folder_path, img_file))
#                     self.labels.append(pid)
#             pid += 1

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         label = self.labels[idx]
#         return image, label


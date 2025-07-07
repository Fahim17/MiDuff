import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoProcessor

#this works version=1.0

class University1652Dataset(Dataset):
    def __init__(self, root_dir, mode='train', num_input=5, num_of_loc = None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'University-1652/')
            mode (str): 'train' or 'test' (we assume paired folders exist under train)
            transform: PyTorch transforms
        """
        
        self.drone_dir = os.path.join(root_dir, mode, 'drone')
        self.satellite_dir = os.path.join(root_dir, mode, 'satellite')
        
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


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
        if it is a single digit (0–9). Otherwise, returns the number as a string.
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


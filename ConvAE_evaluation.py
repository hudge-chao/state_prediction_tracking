import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch 
from models.AE.ConvAE import Encoder, Decoder
import torch.functional as F
import torch.nn as nn
import tensorboardX
import random

class ConvAE_Dataset(Dataset):
    def __init__(self, dir="./maps") -> None:
        super().__init__()
        self.dataset_dir = dir

    def __len__(self) -> int:
        files = os.listdir(self.dataset_dir)
        return len(files)

    def __getitem__(self, index: int):
        image_name = "{}.png".format(index)
        image_path = os.path.join(self.dataset_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        image = np.expand_dims(image, 0)
        return image, index

myDataset = ConvAE_Dataset("./localmaps")

train_dataset_size = int(len(myDataset) * 0.8)
test_dataset_size = len(myDataset) - train_dataset_size

train_dataset, test_dataset = random_split(myDataset, [train_dataset_size, test_dataset_size])

train_dataset_loader = DataLoader(train_dataset, batch_size=20, shuffle=False) 
test_dataset_loader = DataLoader(test_dataset, batch_size=20, shuffle=False) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder()
encoder.load_state_dict(torch.load('weights/ConvAE_weights/300_encoder.pth'), strict=True)
encoder.to(device)

decoder = Decoder()
decoder.load_state_dict(torch.load('weights/ConvAE_weights/300_decoder.pth'), strict=True)
decoder.to(device)

if __name__ == '__main__':
    selected_sample_index = random.randint(0, len(myDataset))
    selected_sample = myDataset[selected_sample_index][0]
    selected_image_origin = np.expand_dims(selected_sample, 0)
    selected_image_origin = torch.tensor(selected_image_origin, dtype=torch.float)
    selected_image_origin_tensor = selected_image_origin.clone().detach().to(device)
    selected_image_infer_tensor = decoder(encoder(selected_image_origin_tensor))
    selected_image_infer = selected_image_infer_tensor.detach().cpu().numpy()[0]
    selected_image_origin_show =  selected_sample[0] * 255.0
    selected_image_infer_show = selected_image_infer[0] * 255.0
    image_concat = np.hstack((selected_image_origin_show, selected_image_infer_show))
    cv2.imwrite("CAE.png", image_concat)
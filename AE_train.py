import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch 
from models.AE.ConvAE import Encoder, Decoder
import torch.functional as F
import torch.nn as nn
import tensorboardX


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

train_dataset = DataLoader(myDataset, batch_size=16, shuffle=False) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('数据集样本数量: ', len(myDataset))

print('图片尺寸: ', myDataset[0][0].shape)

writer = tensorboardX.SummaryWriter('./log', flush_secs=2)

train_epoches = 100

loss_fn = nn.MSELoss()

lr = 0.001

encoder = Encoder()
decoder = Decoder()

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-5)

encoder.to(device)
decoder.to(device)

for i in range(train_epoches):
    train_loss_epoch = []
    for image_batch, index_batch in train_dataset:
        # print(index_batch)
        image_batch_tensor = image_batch.clone().detach().float().to(device)
        encoded_data = encoder(image_batch_tensor)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, image_batch_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch.append(loss.detach().cpu().numpy())
    train_loss_avg = np.mean(train_loss_epoch)
    writer.add_scalar('loss', train_loss_avg, global_step=i)
    print("train epoch: {}, train loss: {}".format(i+1, train_loss_avg))

# torch.save(encoder.state_dict(), "./weights/encoder.pth")
# torch.save(decoder.state_dict(), "./weights/decoder.pth")



# image = myDataset[10]
# image = image.to(device)
# encoded_image = encoder(image)
# output_image = decoder(encoded_image)
# while True:
#     cv2.imshow("input", image)
#     cv2.imshow("output", output_image)
#     cv2.waitKey(10)
        



        
    
    
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
# validate_dataset_size = len(myDataset) - (train_dataset_size + test_dataset_size)

# train_dataset, test_dataset, validate_dataset = random_split(myDataset, [train_dataset_size, test_dataset_size, validate_dataset_size])
train_dataset, test_dataset = random_split(myDataset, [train_dataset_size, test_dataset_size])

# train_dataset_loader = DataLoader(train_dataset, batch_size=20, shuffle=False) 

train_dataset_loader = DataLoader(train_dataset, batch_size=20, shuffle=False) 
test_dataset_loader = DataLoader(test_dataset, batch_size=20, shuffle=False) 
# validate_dataset_loader = DataLoader(validate_dataset, batch_size=20, shuffle=False) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('数据集样本数量: ', len(myDataset))

print('图片尺寸: ', myDataset[0][0].shape)

writer = tensorboardX.SummaryWriter('./log/AE', flush_secs=2)

train_epoches = 500

evaluate_frequency = 10

test_frequency = 5

save_weight_frequency = 50

loss_fn = nn.MSELoss()
# loss_fn = nn.Softmax2d()

lr = 0.005
 
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
    for image_batch, index_batch in train_dataset_loader:
        image_batch_tensor = image_batch.clone().detach().float().to(device)
        encoded_data = encoder(image_batch_tensor)
        decoded_data = decoder(encoded_data)
        loss = torch.sqrt((decoded_data - image_batch_tensor).pow(2).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_numpy = loss.detach().cpu().numpy()
        train_loss_epoch.append(loss_numpy)
    train_loss_avg = np.mean(train_loss_epoch)
    writer.add_scalar('train loss', train_loss_avg * 1000, global_step=i+1)
    print("train epoch: {}, train loss: {}".format(i+1, train_loss_avg * 1000))

    if (i % evaluate_frequency == 0 and i != 0):
        test_loss_epoch = []
        for image_test_batch, index_test_batch in test_dataset_loader:
            image_test_batch_tensor = image_test_batch.clone().detach().float().to(device)
            encoded_test_data = encoder(image_test_batch_tensor)
            decoded_test_data = decoder(encoded_test_data)
            loss = torch.sqrt((decoded_test_data - image_test_batch_tensor).pow(2).mean())
            loss_numpy = loss.detach().cpu().numpy()
            test_loss_epoch.append(loss_numpy)
        test_loss_avg = np.mean(test_loss_epoch)
        writer.add_scalar('test loss', test_loss_avg * 1000, global_step=(i / evaluate_frequency))
        print("test epoch: {}, test loss: {}".format(i / evaluate_frequency, test_loss_avg * 1000))

    if (i % test_frequency == 0 and i != 0):
        print('------test------')
        selected_sample_index = random.randint(0, len(myDataset))
        selected_sample = myDataset[selected_sample_index][0]
        selected_image_origin = np.expand_dims(selected_sample, 0)
        # print(selected_image_origin.shape)
        selected_image_origin = torch.tensor(selected_image_origin, dtype=torch.float)
        selected_image_origin_tensor = selected_image_origin.clone().detach().to(device)
        selected_image_infer_tensor = decoder(encoder(selected_image_origin_tensor))
        selected_image_infer = selected_image_infer_tensor.detach().cpu().numpy()[0]
        selected_image_origin_show =  selected_sample[0] * 255.0
        selected_image_infer_show = selected_image_infer[0] * 255.0
        image_concat = np.hstack((selected_image_origin_show, selected_image_infer_show))
        cv2.imwrite("result.png", image_concat)

    if (i % save_weight_frequency == 0 and i != 0):
        print('======= saving models =======')
        torch.save(encoder.state_dict(), "./weights/{}_encoder.pth".format(i))
        torch.save(decoder.state_dict(), "./weights/{}_decoder.pth".format(i))
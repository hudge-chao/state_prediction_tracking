from pickletools import optimize
from threading import local
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: 
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import csv
import cv2
import numpy as np
import random
import torch 
from models.Tracker.state_track_network import state_predictor
from visualization import visualization_tools
from tensorboardX import SummaryWriter
import torch.nn as nn

writer = SummaryWriter('./predictor_summary_log', flush_secs=1)

class predictor_Dataset(Dataset):
    def __init__(self, dataset_dir:str = 'data') -> None:
        super().__init__()
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        localMaps_dir = os.path.join(self.dataset_dir, 'maps')
        localMaps = os.listdir(localMaps_dir)
        return len(localMaps)

    def __getitem__(self, index: int):
        localMap_fileName = os.path.join(self.dataset_dir + os.sep + 'maps', str(index) + '.png')
        localMap_image = cv2.imread(localMap_fileName, cv2.IMREAD_GRAYSCALE)
        localMap_image = localMap_image / 255.0
        localMap_image = np.expand_dims(localMap_image, 0)
        # localMap_image = localMap_image.astype(np.float)
        path = np.loadtxt(self.dataset_dir + '/path/' + str(index) + '.csv', delimiter=',', dtype=np.float).reshape(-1)
        tracked_postion_img = np.loadtxt(self.dataset_dir + '/tracked_pixel/' + str(index) + '.csv', delimiter=',', dtype=np.float)
        # image_origin = np.loadtxt(self.dataset_dir + '/origin/' + str(index) + '.csv', delimiter=',', dtype=np.float)
        # tracked_position_real = np.loadtxt(self.dataset_dir + '/tracked/' + str(index) + '.csv', delimiter=',', dtype=np.float)
        # tracked_position_img = [int((tracked_position_real[0] - image_origin[0])/0.1), int((tracked_position_real[1] - image_origin[1])/0.1)]
        return localMap_image, path, tracked_postion_img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dataset = predictor_Dataset('./samples')
train_size = int(0.8 * len(state_dataset))
test_size = len(state_dataset) - train_size
train_dataset, test_dataset = random_split(state_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("===== total dataset size: {} =====".format(len(state_dataset)))

print("===== total trainset size: {} =====".format(len(train_dataset)))

print("===== total testset size: {} =====".format(len(test_dataset)))

tracker = state_predictor()

train_epoches = 1000
lr = 0.001
optimizer = torch.optim.Adam(tracker.parameters(), lr=lr)
loss_fn = nn.MSELoss()

test_visualizatin_frequency = 3
test_frequency = 5
save_param_frequency = 20


if __name__ == '__main__':
    tracker.to(device)
    print('======== learning start ========')
    for i in range(train_epoches):
        print('======== training ========')
        train_loss_epoch = []
        for image_batch, trajectory_batch, prediction_batch in train_dataloader:
            image_batch_tensor = image_batch.clone().detach().float().to(device)
            trajectory_batch_tensor = trajectory_batch.detach().clone().float().to(device)
            prediction_batch_tensor = prediction_batch.detach().clone().float().to(device)
            # print(trajectory_batch_tensor.shape)
            output_predictor = tracker.forward(image_batch_tensor, trajectory_batch_tensor)
            loss = loss_fn(prediction_batch_tensor, output_predictor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.detach().cpu().numpy())
        train_loss_avg_epoch = np.mean(train_loss_epoch)
        print('train epoch: {}, train loss: {}'.format(i, train_loss_avg_epoch))
        writer.add_scalar('train/loss', train_loss_avg_epoch, global_step=i)
    
        if (i % test_frequency == 0 and i != 0):
            print('======== testing ========')
            test_loss_epoch = []
            for image_test_batch, trajectory_test_batch, prediction_test_batch in test_dataloader:
                image_test_batch_tensor = image_test_batch.clone().detach().float().to(device)
                trajectory_test_batch_tensor = trajectory_test_batch.clone().detach().float().to(device)
                prediction_test_batch_tensor = prediction_test_batch.clone().detach().float().to(device)
                output_test_predictor = tracker.forward(image_test_batch_tensor, trajectory_test_batch_tensor)
                loss = loss_fn(prediction_test_batch_tensor, output_test_predictor)
                test_loss_epoch.append(loss.detach().cpu().numpy())
            test_loss_avg_epoch = np.mean(test_loss_epoch)
            print('train epoch: {}, test loss: {}'.format(i, test_loss_avg_epoch))
            writer.add_scalar('test/loss', test_loss_avg_epoch, global_step=i)
            
        if (i % test_visualizatin_frequency == 0 and i != 0):
            random_select_index = random.randint(0, len(state_dataset)-1)
            selected_image, selected_trajectory, selected_tracked = state_dataset[random_select_index]
            selected_image = np.expand_dims(selected_image, 0)
            selected_trajectory = np.expand_dims(selected_trajectory, 0)
            selected_image_tensor = torch.from_numpy(selected_image).float().to(device)
            selected_trajectory_tensor = torch.from_numpy(selected_trajectory).float().to(device)
            prediction = tracker.forward(selected_image_tensor, selected_trajectory_tensor)
            prediction = prediction.detach().cpu().numpy()[0]
            selected_image_show = np.squeeze(selected_image)
            selected_trajectory_show = np.squeeze(selected_trajectory)
            selected_trajectory_show = selected_trajectory_show.reshape(30, 2)
            selected_image_show = selected_image_show * 255.0
            visualization_image = visualization_tools(selected_image_show, selected_trajectory_show, selected_tracked, prediction)
            cv2.imwrite("predictor_visual_test.png", visualization_image)

        if (i % save_param_frequency == 0 and i != 0):
            print('========save model========')
            torch.save(tracker.state_dict(), "./weights/state_tracker/{}_predictor.pth".format(i))
    print('======== learning end ========')



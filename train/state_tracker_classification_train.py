from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: 
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import random
import torch 
from models.Tracker.state_tracker_classification import StateTrackerClassification
from visualization import visualization_tools
from tensorboardX import SummaryWriter
import torch.nn as nn

writer = SummaryWriter('log/Tracker/Classification', flush_secs=1)

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
        tracked_postion_img = np.loadtxt(self.dataset_dir + '/tracked_pixel/' + str(index) + '.csv', delimiter=',', dtype=np.int)

        tracked_position = tracked_postion_img[0] * 300 + tracked_postion_img[1]


        # image_origin = np.loadtxt(self.dataset_dir + '/origin/' + str(index) + '.csv', delimiter=',', dtype=np.float)
        # tracked_position_real = np.loadtxt(self.dataset_dir + '/tracked/' + str(index) + '.csv', delimiter=',', dtype=np.float)
        # tracked_position_img = [int((tracked_position_real[0] - image_origin[0])/0.1), int((tracked_position_real[1] - image_origin[1])/0.1)]
        return localMap_image, path, tracked_position


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dataset = predictor_Dataset('./samples')
train_size = int(0.8 * len(state_dataset))
test_size = len(state_dataset) - train_size
train_dataset, test_dataset = random_split(state_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

print("===== total dataset size: {} =====".format(len(state_dataset)))

print("===== total trainset size: {} =====".format(len(train_dataset)))

print("===== total testset size: {} =====".format(len(test_dataset)))

tracker = StateTrackerClassification()

train_epoches = 1000
lr = 0.001
optimizer = torch.optim.Adam(tracker.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

test_visualizatin_frequency = 1
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
            prediction_batch_tensor = prediction_batch.detach().clone().long().to(device)
            output_predictor = tracker.forward(image_batch_tensor, trajectory_batch_tensor)
            # print(output_predictor.shape)
            loss = loss_fn(output_predictor, prediction_batch_tensor)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.detach().cpu().numpy())
        train_loss_avg_epoch = np.sum(train_loss_epoch)
        print('train epoch: {}, train loss: {}'.format(i, train_loss_avg_epoch))
        writer.add_scalar('classification train/loss', train_loss_avg_epoch, global_step=i)
    
        if (i % test_frequency == 0 and i != 0):
            print('======== testing ========')
            test_loss_epoch = []
            for image_test_batch, trajectory_test_batch, prediction_test_batch in test_dataloader:
                image_test_batch_tensor = image_test_batch.clone().detach().float().to(device)
                trajectory_test_batch_tensor = trajectory_test_batch.clone().detach().float().to(device)
                prediction_test_batch_tensor = prediction_test_batch.clone().detach().long().to(device)
                output_test_predictor = tracker.forward(image_test_batch_tensor, trajectory_test_batch_tensor)
                # output_test_predictor = torch.softmax(output_test_predictor, 1)
                loss = loss_fn(output_test_predictor, prediction_test_batch_tensor)
                test_loss_epoch.append(loss.detach().cpu().numpy())
            test_loss_avg_epoch = np.sum(test_loss_epoch)
            print('train epoch: {}, test loss: {}'.format(i, test_loss_avg_epoch))
            writer.add_scalar('classification test/loss', test_loss_avg_epoch, global_step=i)
            
        if (i % test_visualizatin_frequency == 0):
            random_select_index = random.randint(0, len(state_dataset)-1)
            selected_image, selected_trajectory, selected_tracked = state_dataset[random_select_index]
            selected_image = np.expand_dims(selected_image, 0)
            selected_trajectory = np.expand_dims(selected_trajectory, 0)
            selected_image_tensor = torch.from_numpy(selected_image).float().to(device)
            selected_trajectory_tensor = torch.from_numpy(selected_trajectory).float().to(device)
            prediction = tracker.forward(selected_image_tensor, selected_trajectory_tensor)
            prediction = torch.softmax(prediction, 1)
            prediction = prediction.detach().cpu().numpy()[0]
            prediction_index = np.unravel_index(np.ma.argmax(prediction), (300, 300))
            prediction_index_show = np.array([prediction_index[0], prediction_index[1]])
            selected_tracked_index = np.unravel_index(selected_tracked, (300, 300))
            selected_tracked_index_show = np.array([selected_tracked_index[0], selected_tracked_index[1]])
            print(prediction_index)
            selected_image_show = np.squeeze(selected_image)
            selected_trajectory_show = np.squeeze(selected_trajectory)
            selected_trajectory_show = selected_trajectory_show.reshape(30, 2)
            selected_image_show = selected_image_show * 255.0
            visualization_image = visualization_tools(selected_image_show, selected_trajectory_show, selected_tracked_index_show, prediction_index_show)
            cv2.imwrite("predictor_visual_test.png", visualization_image)

        if (i % save_param_frequency == 0 and i != 0):
            print('========save model========')
            torch.save(tracker.state_dict(), "./weights/state_tracker_union/{}_tracker_classify.pth".format(i))
    print('======== learning end ========')



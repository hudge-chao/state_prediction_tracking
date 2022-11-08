import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path: 
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.Tracker.state_track_trajectory_only import StateTrackerTrajectory
from tensorboardX import SummaryWriter
import torch.nn as nn

writer = SummaryWriter('log/Tracker/trajectory_only', flush_secs=1)

class TrackerDataset(Dataset):
    def __init__(self, dataset_dir:str = 'data') -> None:
        super().__init__()
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        trajectory_dir = os.path.join(self.dataset_dir, 'trajectory')
        trajectory_files = os.listdir(trajectory_dir)
        return len(trajectory_files)

    def __getitem__(self, index: int):
        trajectory = np.loadtxt(self.dataset_dir + '/trajectory/' + str(index) + '.txt', delimiter=',', dtype=np.float).reshape(-1)
        tracked_position = np.loadtxt(self.dataset_dir + '/tracked/' + str(index) + '.csv', delimiter=',', dtype=np.float)
        return trajectory, tracked_position

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dataset = TrackerDataset('./samples')
print(state_dataset)
train_size = int(0.8 * len(state_dataset))
test_size = len(state_dataset) - train_size
train_dataset, test_dataset = random_split(state_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("===== total dataset size: {} =====".format(len(state_dataset)))

print("===== total trainset size: {} =====".format(len(train_dataset)))

print("===== total testset size: {} =====".format(len(test_dataset)))

tracker = StateTrackerTrajectory()

train_epoches = 1000
lr = 0.001
optimizer = torch.optim.Adam(tracker.parameters(), lr=lr)
loss_fn = nn.MSELoss()

test_frequency = 5

save_param_frequency = 20

if __name__ == '__main__':
    tracker.to(device)
    print('======== learning start ========')
    for i in range(train_epoches):
        print('======== training ========')
        train_loss_epoch = []
        for trajectory_batch, prediction_batch in train_dataloader:
            trajectory_batch_tensor = trajectory_batch.detach().clone().float().to(device)
            prediction_batch_tensor = prediction_batch.detach().clone().float().to(device)
            output_predictor = tracker.forward(trajectory_batch_tensor)
            loss = loss_fn(prediction_batch_tensor, output_predictor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.detach().cpu().numpy())
        train_loss_avg_epoch = np.mean(train_loss_epoch)
        print('train epoch: {}, train loss: {}'.format(i, train_loss_avg_epoch))
        writer.add_scalar('trajectory tracker train/loss', train_loss_avg_epoch, global_step=i+1)
    
        if (i % test_frequency == 0 and i != 0):
            print('======== testing ========')
            test_loss_epoch = []
            for trajectory_test_batch, prediction_test_batch in test_dataloader:
                trajectory_test_batch_tensor = trajectory_test_batch.clone().detach().float().to(device)
                prediction_test_batch_tensor = prediction_test_batch.clone().detach().float().to(device)
                output_test_predictor = tracker.forward(trajectory_test_batch_tensor)
                loss = loss_fn(prediction_test_batch_tensor, output_test_predictor)
                test_loss_epoch.append(loss.detach().cpu().numpy())
            test_loss_avg_epoch = np.mean(test_loss_epoch)
            print('train epoch: {}, test loss: {}'.format(i+1, test_loss_avg_epoch))
            writer.add_scalar('trajectory tracker test/loss', test_loss_avg_epoch, global_step=i+1)

        if (i % save_param_frequency == 0 and i != 0):
            print('========save model========')
            torch.save(tracker.state_dict(), "./weights/state_tracker_single/{}_tracker.pth".format(i))
    print('======== learning end ========')
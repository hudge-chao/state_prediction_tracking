from collections import OrderedDict
import torch
import torch.nn as nn

class state_predictor(nn.Module):
    def __init__(self, input_dims=6400, hidden_dims=640, output_dims=2, model : str = 'train') -> None:
        super(state_predictor, self).__init__()

        # CNNs : input b * 300 * 300 * 2
        self.map_conv_net = nn.Sequential(OrderedDict([
            ('map_conv_net-conv0', nn.Conv2d(2, 4, 3, 1, 0)),
            ('map_conv_net-norm0', nn.BatchNorm2d(4)),
            ('map_conv_net-relu0', nn.ReLU(True)), # (4, 298, 298)

            ('map_conv_net-conv1', nn.Conv2d(4, 8, 3, 1, 0)), 
            ('map_conv_net-norm1', nn.BatchNorm2d(8)),
            ('map_conv_net-relu1', nn.ReLU(True)), # (8, 296, 296)

            ('map_conv_net-conv2', nn.Conv2d(8, 16, 3, 1, 0)),
            ('map_conv_net-norm2', nn.BatchNorm2d(16)),
            ('map_conv_net-relu2', nn.ReLU(True)), # (16, 294, 294)

            ('map_conv_net-pool0', nn.MaxPool2d(2, 2, 0)), # (16, 147, 147)

            ('map_conv_net-conv3', nn.Conv2d(16, 32, 2, 1, 0)),
            ('map_conv_net-norm3', nn.BatchNorm2d(32)),
            ('map_conv_net-relu3', nn.ReLU(True)), # (32, 146, 146)

            ('map_conv_net-conv4', nn.Conv2d(32, 32, 3, 1, 0)),
            ('map_conv_net-norm4', nn.BatchNorm2d(32)),
            ('map_conv_net-relu4', nn.ReLU(True)), # (32, 144, 144)

            ('map_conv_net-pool1', nn.MaxPool2d(2, 2, 0)), # (32, 72, 72)

            ('map_conv_net-conv5', nn.Conv2d(32, 64, 3, 1, 0)),
            ('map_conv_net-norm5', nn.BatchNorm2d(64)),
            ('map_conv_net-relu5', nn.ReLU(True)), # (64, 70, 70)

            ('map_conv_net-conv6', nn.Conv2d(64, 64, 3, 1, 0)),
            ('map_conv_net-norm6', nn.BatchNorm2d(64)),
            ('map_conv_net-relu6', nn.ReLU(True)), # (64, 68, 68)

            ('map_conv_net-pool2', nn.MaxPool2d(2, 2, 0)), # (64, 34, 34)

            ('map_conv_net-conv7', nn.Conv2d(64, 64, 3, 1, 0)),
            ('map_conv_net-norm7', nn.BatchNorm2d(64)),
            ('map_conv_net-relu7', nn.ReLU(True)), # (64, 32, 32)

            ('map_conv_net-conv8', nn.Conv2d(64, 128, 3, 1, 0)),
            ('map_conv_net-norm8', nn.BatchNorm2d(128)),
            ('map_conv_net-relu8', nn.ReLU(True)), # (128, 30, 30)

            ('map_conv_net-conv9', nn.Conv2d(128, 128, 3, 1, 0)),
            ('map_conv_net-norm9', nn.BatchNorm2d(128)),
            ('map_conv_net-relu9', nn.ReLU(True)), # (128, 28, 28)

            ('map_conv_net-pool3', nn.MaxPool2d(2, 2, 0)), # (128, 14, 14)

            ('map_conv_net-conv10', nn.Conv2d(128, 256, 3, 1, 0)),
            ('map_conv_net-norm10', nn.BatchNorm2d(256)),
            ('map_conv_net-relu10', nn.ReLU(True)), # (256, 12, 12)

            ('map_conv_net-conv11', nn.Conv2d(256, 256, 3, 1, 0)),
            ('map_conv_net-norm11', nn.BatchNorm2d(256)),
            ('map_conv_net-relu11', nn.ReLU(True)), # (256, 10, 10)

            ('map_conv_net-pool4', nn.MaxPool2d(2, 2, 0)) # (256, 5, 5)
        ]))
        
        # 256 * 5 * 5 = 6400 
        self.predictor_net = nn.Sequential(OrderedDict([
            ('predictor-line0', nn.Linear(input_dims, hidden_dims*10)),
            ('predictor-relu0', nn.ReLU(True)),
            ('predictor-line1', nn.Linear(hidden_dims*10, hidden_dims*8)),
            ('predictor-relu1', nn.ReLU(True)),
            ('predictor-line2', nn.Linear(hidden_dims*8, hidden_dims*4)),
            ('predictor-relu2', nn.ReLU(True)),
            ('predictor-line3', nn.Linear(hidden_dims*4, hidden_dims*2)),
            ('predictor-relu3', nn.ReLU(True)),
            ('predictor-line4', nn.Linear(hidden_dims*2, hidden_dims)),
            ('predictor-relu4', nn.ReLU(True)),
            ('predictor-line5', nn.Linear(hidden_dims, output_dims))
        ]))

    def forward(self, multi_union_input):
        # map_encoded = self.encoder.forward(local_map)
        map_feat = self.map_conv_net(multi_union_input)
        map_feat_vector = map_feat.view(map_feat.size(0), -1)
        # features_union = torch.cat((map_feat_vector, trajectory), dim=1)
        outputs = self.predictor_net(map_feat_vector)
        return outputs

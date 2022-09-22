from collections import OrderedDict
import torch
import torch.nn as nn
from models.AE.ConvAE import Encoder

class StateTrackerClassification(nn.Module):
    def __init__(self, input_dims=124, output_dims=300*300, model : str = 'train') -> None:
        super(StateTrackerClassification, self).__init__()

        # use encoder 
        self.encoder = Encoder()

        if model == 'train':
            self.encoder.load_state_dict(torch.load('./weights/ConvAE_weights/encoder.pth'), strict=True)

            for name, param in self.encoder.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False

        self.map_conv_net = nn.Sequential(OrderedDict([
            ('map_conv_net-conv0', nn.Conv2d(64, 32, 1, 1, 0)),
            ('map_conv_net-norm0', nn.BatchNorm2d(32)),
            ('map_conv_net-relu0', nn.ReLU(True)), # (32, 10, 10)

            ('map_conv_net-conv1', nn.Conv2d(32, 16, 2, 1, 0)),
            ('map_conv_net-norm1', nn.BatchNorm2d(16)),
            ('map_conv_net-relu1', nn.ReLU(True)), # (16, 9, 9)

            ('map_conv_net-conv2', nn.Conv2d(16, 8, 2, 1, 0)),
            ('map_conv_net-norm2', nn.BatchNorm2d(8)),
            ('map_conv_net-relu2', nn.ReLU(True)), # (8, 8, 8)

            ('map_conv_net-pool0', nn.MaxPool2d(2, 2, 0)), # (8, 4, 4)

            ('map_conv_net-conv3', nn.Conv2d(8, 8, 1, 1, 0)),
            ('map_conv_net-norm3', nn.BatchNorm2d(8)),
            ('map_conv_net-relu3', nn.ReLU(True)), # (8, 4, 4)

            ('map_conv_net-conv4', nn.Conv2d(8, 4, 1, 1, 0)),
            ('map_conv_net-norm4', nn.BatchNorm2d(4)),
            ('map_conv_net-relu4', nn.ReLU(True)), # (4, 4, 4)
        ]))
        

        # 4*4*4 + 60 = 64 + 60 = 124
        self.predictor_net = nn.Sequential(OrderedDict([
            ('predictor-line0', nn.Linear(input_dims, output_dims)),
            # ('predictor-relu0', nn.ReLU(True)),
            # ('predictor-line1', nn.Linear(8192, 16384)),
            # ('predictor-relu1', nn.ReLU(True)),
            # ('predictor-line2', nn.Linear(16384, output_dims))
        ]))

        for m in self.named_modules():
            if 'map_conv_net-' in m[0] or 'predictor-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.Linear):
                    m[1].weight.data.normal_(0, 1e-3)
                    m[1].bias.data.zero_()
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, local_map, trajectory):
        map_encoded = self.encoder.forward(local_map)
        map_feat = self.map_conv_net(map_encoded)
        map_feat_vector = map_feat.view(map_feat.size(0), -1)
        features_union = torch.cat((map_feat_vector, trajectory), dim=1)
        outputs = self.predictor_net(features_union)
        return outputs

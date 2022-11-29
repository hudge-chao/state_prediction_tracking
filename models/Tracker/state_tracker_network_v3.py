from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import densenet121
from models.AE.ConvAE import Encoder

class state_predictor(nn.Module):
    def __init__(self, input_dims=124, hidden_dims=32, output_dims=2, load_mode : str = 'train') -> None:
        super(state_predictor, self).__init__()

        self.encoder_cnn = nn.Sequential(OrderedDict([
            ('encoder-conv0', nn.Conv2d(2, 8, 2, stride=1)),
            ('encoder-norm0', nn.BatchNorm2d(8)),
            ('encoder-relu0', nn.ReLU(True)), # (8, 299, 299)

            ('encoder-conv1', nn.Conv2d(8, 8, 2, stride=1)),
            ('encoder-norm1', nn.BatchNorm2d(8)),
            ('encoder-relu1', nn.ReLU(True)), # (8, 298, 298)

            ('encoder-pool0', nn.MaxPool2d(2, 2, 0)), # (8, 149, 149)

            ('encoder-conv2', nn.Conv2d(8, 16, 2, stride=1)),
            ('encoder-norm2', nn.BatchNorm2d(16)),
            ('encoder-relu2', nn.ReLU(True)), # (16, 148, 148)

            ('encoder-conv3', nn.Conv2d(16, 16, 2, stride=1)),
            ('encoder-norm3', nn.BatchNorm2d(16)),
            ('encoder-relu3', nn.ReLU(True)), # (16, 147, 147)

            ('encoder-conv4', nn.Conv2d(16, 16, 2, stride=1)),
            ('encoder-norm4', nn.BatchNorm2d(16)),
            ('encoder-relu4', nn.ReLU(True)), # (16, 146, 146)

            ('encoder-pool1', nn.MaxPool2d(2, 2, 0)), # (16, 73, 73)

            ('encoder-conv5', nn.Conv2d(16, 32, 2, stride=1)),
            ('encoder-norm5', nn.BatchNorm2d(32)),
            ('encoder-relu5', nn.ReLU(True)), # (32, 72, 72)

            ('encoder-conv6', nn.Conv2d(32, 32, 2, stride=1)),
            ('encoder-norm6', nn.BatchNorm2d(32)),
            ('encoder-relu6', nn.ReLU(True)), # (32, 71, 71)

            ('encoder-conv7', nn.Conv2d(32, 32, 2, stride=1)),
            ('encoder-norm7', nn.BatchNorm2d(32)),
            ('encoder-relu7', nn.ReLU(True)), # (32, 70, 70)

            ('encoder-pool2', nn.MaxPool2d(2, 2, 0)), # (32, 35, 35)

            ('encoder-conv8', nn.Conv2d(32, 64, 2, stride=1)),
            ('encoder-norm8', nn.BatchNorm2d(64)),
            ('encoder-relu8', nn.ReLU(True)), # (64, 34, 34)

            ('encoder-conv9', nn.Conv2d(64, 64, 2, stride=1)),
            ('encoder-norm9', nn.BatchNorm2d(64)),
            ('encoder-relu9', nn.ReLU(True)), # (64, 33, 33)

            ('encoder-conv10', nn.Conv2d(64, 64, 2, stride=1)),
            ('encoder-norm10', nn.BatchNorm2d(64)),
            ('encoder-relu10', nn.ReLU(True)), # (64, 32, 32)

            ('encoder-pool3', nn.MaxPool2d(2, 2, 0)), # (64, 16, 16)

            ('encoder-conv11', nn.Conv2d(64, 64, 3, stride=1)),
            ('encoder-norm11', nn.BatchNorm2d(64)),
            ('encoder-relu11', nn.ReLU(True)), # (64, 14, 14)

            ('encoder-conv12', nn.Conv2d(64, 64, 3, stride=1)),
            ('encoder-norm12', nn.BatchNorm2d(64)),
            ('encoder-relu12', nn.ReLU(True)), # (64, 12, 12)

            ('encoder-conv13', nn.Conv2d(64, 64, 3, stride=1)),
            ('encoder-norm13', nn.BatchNorm2d(64)),
            ('encoder-relu13', nn.ReLU(True)), # (64, 10, 10)
        ]))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 12, 12)

            nn.ConvTranspose2d(64, 64, 3, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 14, 14)

            nn.ConvTranspose2d(64, 64, 3, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 16, 16)

            # First transposed convolution
            nn.Upsample(scale_factor=2, mode='nearest'), # (64, 32, 32)
            # nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(64, 64, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 33, 33)

            nn.ConvTranspose2d(64, 64, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 34, 34)

            nn.ConvTranspose2d(64, 32, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True), # (32, 35, 35)

            nn.Upsample(scale_factor=2, mode='nearest'),  # (32, 70, 70)
            # nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(32, 32, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True), # (32, 71, 71)

            nn.ConvTranspose2d(32, 32, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True), # (32, 72, 72)

            nn.ConvTranspose2d(32, 16, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True), # (16, 73, 73)

            nn.Upsample(scale_factor=2, mode='nearest'), # (16, 146, 146)
            # nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(16, 16, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True), # (16, 147, 147)

            nn.ConvTranspose2d(16, 16, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True), # (16, 148, 148)

            nn.ConvTranspose2d(16, 8, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(True), # (8, 149, 149)

            nn.Upsample(scale_factor=2, mode='nearest'), # (8, 298, 298)
            # nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(8, 8, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(True), # (8, 299, 299)

            nn.ConvTranspose2d(8, 1, 2, stride=1, output_padding=(0, 0)),
            # nn.BatchNorm2d(1),
            # nn.ReLU(True), # (1, 300, 300)
        )

        self.mode = load_mode

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
            ('predictor-line0', nn.Linear(input_dims, hidden_dims*8)),
            ('predictor-relu0', nn.ReLU(True)),
            ('predictor-line1', nn.Linear(hidden_dims*8, hidden_dims*4)),
            ('predictor-relu1', nn.ReLU(True)),
            ('predictor-line2', nn.Linear(hidden_dims*4, hidden_dims*4)),
            ('predictor-relu2', nn.ReLU(True)),
            ('predictor-line3', nn.Linear(hidden_dims*4, hidden_dims*2)),
            ('predictor-relu3', nn.ReLU(True)),
            ('predictor-line4', nn.Linear(hidden_dims*2, hidden_dims)),
            ('predictor-relu4', nn.ReLU(True)),
            ('predictor-line5', nn.Linear(hidden_dims, output_dims))
        ]))

    def forward(self, union_image):
        # map_encoded = self.encoder.forward(local_map)
        # map_feat = self.map_conv_net(map_encoded)
        # map_feat_vector = map_feat.view(map_feat.size(0), -1)
        # features_union = torch.cat((map_feat_vector, trajectory), dim=1)
        # outputs = self.predictor_net(features_union)
        # outputs = self.feat_conv_net(local_map)
        encoder_outputs = self.encoder_cnn(union_image)
        outputs = self.decoder_conv(encoder_outputs)
        return outputs

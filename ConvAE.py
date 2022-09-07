from collections import OrderedDict
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, channel=32, width=27, height=27):
        super().__init__()
        self.channel = channel
        self.width = width
        self.height = height

    def forward(self, x):
        x = x.reshape(x.shape[0], self.channel, self.width, self.height)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_cnn = nn.Sequential(OrderedDict([
            ('encoder-conv0', nn.Conv2d(1, 8, 2, stride=1)),
            ('encoder-norm0', nn.BatchNorm2d(8)),
            ('encoder-relu0', nn.ReLU(True)), # (8, 199, 199)

            ('encoder-conv1', nn.Conv2d(8, 8, 2, stride=1)),
            ('encoder-norm1', nn.BatchNorm2d(8)),
            ('encoder-relu1', nn.ReLU(True)), # (8, 198, 198)

            ('encoder-pool0', nn.MaxPool2d(2, 2, 0)), # (8, 99, 99)

            ('encoder-conv2', nn.Conv2d(8, 16, 2, stride=1)),
            ('encoder-norm2', nn.BatchNorm2d(16)),
            ('encoder-relu2', nn.ReLU(True)), # (16, 98, 98)

            ('encoder-conv3', nn.Conv2d(16, 16, 2, stride=1)),
            ('encoder-norm3', nn.BatchNorm2d(16)),
            ('encoder-relu3', nn.ReLU(True)), # (16, 97, 97)

            ('encoder-conv4', nn.Conv2d(16, 16, 2, stride=1)),
            ('encoder-norm4', nn.BatchNorm2d(16)),
            ('encoder-relu4', nn.ReLU(True)), # (16, 96, 96)

            ('encoder-pool1', nn.MaxPool2d(2, 2, 0)), # (16, 48, 48)

            ('encoder-conv5', nn.Conv2d(16, 32, 2, stride=1)),
            ('encoder-norm5', nn.BatchNorm2d(32)),
            ('encoder-relu5', nn.ReLU(True)), # (16, 47, 47)

            ('encoder-conv6', nn.Conv2d(32, 32, 2, stride=1)),
            ('encoder-norm6', nn.BatchNorm2d(32)),
            ('encoder-relu6', nn.ReLU(True)), # (16, 46, 46)

            ('encoder-pool2', nn.MaxPool2d(2, 2, 0)), # (32, 23, 23)

            ('encoder-conv7', nn.Conv2d(32, 64, 2, stride=1)),
            ('encoder-norm7', nn.BatchNorm2d(64)),
            ('encoder-relu7', nn.ReLU(True)), # (64, 22, 22)

            ('encoder-conv8', nn.Conv2d(64, 64, 2, stride=1)),
            ('encoder-norm8', nn.BatchNorm2d(64)),
            ('encoder-relu8', nn.ReLU(True)), # (64, 21, 21)

            ('encoder-conv9', nn.Conv2d(64, 64, 2, stride=1)),
            ('encoder-norm9', nn.BatchNorm2d(64)),
            ('encoder-relu9', nn.ReLU(True)), # (64, 20, 20)

            ('encoder-pool3', nn.MaxPool2d(2, 2, 0)), # (64, 10, 10)
        ]))

        # self.encoder_flatten = Flatten()

        # self.encoder_linear = nn.Sequential(
        #     nn.Linear(32*24*24, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 64)
        # )

    def forward(self, x):
        vec = self.encoder_cnn(x)
        # flatten_feats = self.encoder_flatten(features)
        # vec = self.encoder_linear(flatten_feats)
        return vec


class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()

        ### Linear section
        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(64, 256), 
        #     nn.ReLU(True),
        #     nn.Linear(256, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 32*24*24),
        #     nn.ReLU(True)
        # )

        ### Unflatten
        # self.decoder_unflatten = UnFlatten(channel=32, width=24, height=24)

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.Upsample(scale_factor=2, mode='nearest'), # (64, 20, 20)

            nn.ConvTranspose2d(64, 64, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 21, 21)

            nn.ConvTranspose2d(64, 64, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # (64, 22, 22)

            nn.ConvTranspose2d(64, 32, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True), # (32, 23, 23)

            nn.Upsample(scale_factor=2, mode='nearest'),  # (32, 46, 46)

            nn.ConvTranspose2d(32, 32, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(True), # (16, 47, 47)

            nn.ConvTranspose2d(32, 16, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True), # (16, 48, 48)

            nn.Upsample(scale_factor=2, mode='nearest'), # (16, 96, 96)

            nn.ConvTranspose2d(16, 16, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True), # (16, 97, 97)

            nn.ConvTranspose2d(16, 16, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(True), # (16, 98, 98)

            nn.ConvTranspose2d(16, 8, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(True), # (8, 99, 99)

            nn.Upsample(scale_factor=2, mode='nearest'), # (8, 198, 198)

            nn.ConvTranspose2d(8, 8, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(True), # (8, 199, 199)

            nn.ConvTranspose2d(8, 1, 2, stride=1, output_padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(True), # (1, 200, 200)
        )
        
    def forward(self, x):
        # Apply linear layers
        # x = self.decoder_lin(x)
        # Unflatten
        # x = self.decoder_unflatten(x)
        # # Apply transposed convolutions
        x = self.decoder_conv(x)
        # # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        # x = torch.sigmoid(x)
        return x


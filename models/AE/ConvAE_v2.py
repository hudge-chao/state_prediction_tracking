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
            nn.BatchNorm2d(1),
            nn.ReLU(True), # (1, 300, 300)
        )
        
    def forward(self, x):
        x = self.decoder_conv(x)
        return x


# ========================================================
#             Media and Cognition
#             Homework 2 Convolutional Neural Network
#             networks.py - Network definition
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_batch_norm=False,
        use_residual=False,
    ):
        """
        Convolutional block with batch normalization and ReLU activation
        ----------------------
        :param in_channels: channel number of input image
        :param out_channels: channel number of output image
        :param kernel_size: size of convolutional kernel
        :param stride: stride of convolutional operation
        :param padding: padding of convolutional operation
        :param use_batch_norm: whether to use batch normalization in convolutional layers
        :param use_residual: whether to use residual connection
        """
        super().__init__()

        if use_batch_norm:
            bn2d = nn.BatchNorm2d
        else:
            # use identity function to replace batch normalization
            bn2d = nn.Identity

        self.use_residual = use_residual

        # >>> TODO 2.1: complete a convolutional block with batch normalization and ReLU activation
        # Hint: use the `bn2d` defined above for batch normalization to adapt to the input parameter `use_batch_norm`
        # Network structure:
        # conv -> batchnorm -> relu
        # self.conv = 
        # self.bn = 
        # self.relu = 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = bn2d(out_channels)
        self.relu = nn.ReLU()

        # <<< TODO 2.1

    def forward(self, x):
        # >>> TODO 2.2: forward process
        # Hint: apply residual connection if `self.use_residual` is True
        # # Step 1: apply convolutional operation
        # out = self.conv(x)
        # # Step 2: apply batch normalization
        # out = self.bn(out)
        # # Step 3: apply ReLU activation
        # out = self.relu(out)
        out=self.relu(self.bn(self.conv(x)))
        # Step 4: apply residual connection
        if self.use_residual:
            out = out + x
        # <<< TODO 2.2
        return out


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        use_batch_norm=False,
        use_stn=False,
        dropout_prob=0,
    ):
        """
        Convolutional Neural Networks
        ----------------------
        :param in_channels: channel number of input image
        :param num_classes: number of classes for the classification task
        :param use_batch_norm: whether to use batch normalization in convolutional layers and linear layers
        :param use_stn: whether to use spatial transformer network
        :param dropout_prob: dropout ratio of dropout layer which ranges from 0 to 1
        """
        super().__init__()

        if use_batch_norm:
            bn1d = nn.BatchNorm1d
        else:
            # use identity function to replace batch normalization
            bn1d = nn.Identity

        if use_stn:
            self.stn = STN(in_channels)
        else:
            # use identity function to replace spatial transformer network
            self.stn = nn.Identity(in_channels)

        # >>> TODO 3.1: complete a multilayer convolutional neural network with nn.Sequential function.
        # input image with size [batch_size, in_channels, img_h, img_w]
        # Network structure:
        #            kernel_size  stride  padding  out_channels  use_residual
        # ConvBlock       5          1        2          32         False
        # ConvBlock       5          2        2          64         False
        # maxpool         2          2        0
        # ConvBlock       3          1        1          64         True
        # ConvBlock       3          1        1          128        False
        # maxpool         2          2        0
        # ConvBlock       3          1        1          128        True
        # dropout(p), where p is input parameter of dropout ratio

        # self.conv_net = nn.Sequential()
        self.conv_net = nn.Sequential(
            ConvBlock(in_channels, 32, 5, 1, 2, use_batch_norm,use_residual=False),
            ConvBlock(32, 64, 5, 2, 2, use_batch_norm,use_residual=False),
            nn.MaxPool2d(2, 2, 0),
            ConvBlock(64, 64, 3, 1, 1, use_batch_norm,use_residual=True),
            ConvBlock(64, 128, 3, 1, 1, use_batch_norm, use_residual=False),
            nn.MaxPool2d(2, 2, 0),
            ConvBlock(128, 128, 3, 1, 1, use_batch_norm, use_residual=True),
            nn.Dropout(dropout_prob)
        )

        
        # <<< TODO 3.1

        # >>> TODO 3.2: complete a sub-network with two linear layers by using nn.Sequential function
        # Hint:
        #   (1) Note that the size of input images is (3, 32, 32) by default, what is the size of
        #       the output of the convolution layers?
        #   (2) Use the `bn1d` defined above for batch normalization to adapt to the input parameter `use_batch_norm`
        # Network structure:
        #            out_channels
        # linear          256
        # activation
        # batchnorm
        # dropout(p), where p is input parameter of dropout ratio
        # linear       num_classes
        self.fc_net = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            bn1d(256),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes),
            
         
        )
        # <<< TODO 3.2

    def forward(self, x):
        """
        Define the forward function
        :param x: input features with size [batch_size, in_channels, img_h, img_w]
        :return: output features with size [batch_size, num_classes]
        """
        # Step 1: apply spatial transformer network if applicable
        x = self.stn(x)

        # >>> TODO 3.3: forward process
        # Step 2: forward process for the convolutional network
        out = self.conv_net(x)


        # Step 3: use `Tensor.view()` to flatten the tensor to match the size of the input of the
        # fully connected layers.
        out = out.view(out.size(0), -1)


        # Step 4: forward process for the fully connected network
        out = self.fc_net(out)


        # <<< TODO 3.3

        return out


class STN(nn.Module):
    def __init__(self, in_channels):
        """
        The spatial transformer network (STN) learns how to perform spatial transformations on the
        input image in order to enhance the geometric invariance of the model. For example, it can
        crop a region of interest, scale and correct the orientation of an image. It can be a useful
        mechanism because CNNs are not invariant to rotation and scale and more general affine
        transformations.

        The spatial transformer network boils down to three main components:

        - The localization network is a regular CNN which regresses the transformation parameters.
          The transformation is never learned explicitly from this dataset, instead the network
          learns automatically the spatial transformations that enhances the global accuracy.
        - The grid generator generates a grid of coordinates in the input image corresponding
          to each pixel from the output image.
        - The sampler uses the parameters of the transformation and applies it to the input image.

        Here, we are going to implement an STN that performs affine transformations on the input images.
        For more information, please refer to the slides and
        https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html .

        ----------------------
        :param in_channels: channel number of input image
        """
        super().__init__()

        # >>> TODO 4.1: Build your localization net
        # Step 1: Build a convolutional network to extract features from input images.
        # Hint: Combine convolutional layers, batch normalization layers and ReLU activation functions to build
        # this network.
        # Suggested structure: 3 down-sampling convolutional layers with doubling output channels, using BN and ReLU.
        self.localization_conv = nn.Sequential(
            ConvBlock(in_channels, 32, 3, 1, 1, True, False),
            ConvBlock(32, 64, 3, 2, 1, True, False),
            ConvBlock(64, 128, 3, 2, 1, True, False),
        )


        # Step 2: Build a fully connected network to predict the parameters of affine transformation from
        # the extracted features.
        # Hint: Combine linear layers and ReLU activation functions to build this network.
        # Suggested structure: 2 linear layers with one BN and ReLU.
        self.localization_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        # <<< TODO 4.1

        # >>> TODO 4.2: Initialize the weight/bias of the last linear layer of the fully connected network
        # Hint: The STN should generate the identity transformation by default before training.
        # How to initialize the weight/bias of the last linear layer of the fully connected network to
        # achieve this goal?
        # Initialize the last linear layer to predict the identity transformation
        self.localization_fc[-1].weight.data.zero_()
        self.localization_fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


        # <<< TODO 4.2

    def forward(self, x):
        # Extract the features from input images and flatten them
        features = self.localization_conv(x)
        features = features.view(features.shape[0], -1)

        # Predict the parameters of affine transformation from the extracted features
        theta = self.localization_fc(features)
        theta = theta.view(-1, 2, 3)

        # Apply affine transformation to input images
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

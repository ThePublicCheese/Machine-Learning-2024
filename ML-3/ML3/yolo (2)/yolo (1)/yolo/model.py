"""
CS 6375 Homework 2 Programming
Implement the create_modules() function in this python script
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np


# the YOLO network class
class YOLO(nn.Module):
    def __init__(self, num_boxes, num_classes):
        super(YOLO, self).__init__()
        # number of bounding boxes per cell (2 in our case)
        self.num_boxes = num_boxes
        # number of classes for detection (1 in our case: cracker box)
        self.num_classes = num_classes
        self.image_size = 448
        self.grid_size = 64
        # create the network
        self.network = self.create_modules()
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    #TODO: implement this function to build the network
    def create_modules(self):
        modules = nn.Sequential()

        #16 x 448 x 448
        modules.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu1", nn.ReLU())
        
        #16 x 224 x 224
        modules.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))

        #32 x 224 x 224
        modules.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu2", nn.ReLU())
        
        #32 x 112 x 112
        modules.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))

        #64 x 112 x 112
        modules.add_module("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu3", nn.ReLU())
        
        #64 x 56 x 56
        modules.add_module("pool3", nn.MaxPool2d(kernel_size=2, stride=2))

        #128 x 56 x 56
        modules.add_module("conv4", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu4", nn.ReLU())
        
        #128 x 28 x 28
        modules.add_module("pool4", nn.MaxPool2d(kernel_size=2, stride=2))

        #256 x 28 x 28
        modules.add_module("conv5", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu5", nn.ReLU())
        
        #256 x 14 x 14
        modules.add_module("pool5", nn.MaxPool2d(kernel_size=2, stride=2))

        #512 x 14 x 14
        modules.add_module("conv6", nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu6", nn.ReLU())
        
        #512 x 7 x 7
        modules.add_module("pool6", nn.MaxPool2d(kernel_size=2, stride=2))

        #1024 x 7 x 7
        modules.add_module("conv7", nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu7", nn.ReLU())
        modules.add_module("conv8", nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu8", nn.ReLU())
        modules.add_module("conv9", nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        modules.add_module("relu9", nn.ReLU())

        #50176
        modules.add_module("flatten", nn.Flatten())

        #256
        modules.add_module("fc1", nn.Linear(1024 * 7 * 7, 256))
        modules.add_module("fc2", nn.Linear(256, 7 * 7 * (5 * self.num_boxes + self.num_classes)))

        #7 x 7 x (5b + C)
        modules.add_module("sigmoid", nn.Sigmoid())

        return modules



    # output (batch_size, 5*B + C, 7, 7)
    # In the network output (cx, cy, w, h) are normalized to be [0, 1]
    # This function undo the noramlization to obtain the bounding boxes in the orignial image space
    def transform_predictions(self, output):
        batch_size = output.shape[0]
        x = torch.linspace(0, 384, steps=7)
        y = torch.linspace(0, 384, steps=7)
        corner_x, corner_y = torch.meshgrid(x, y, indexing='xy')
        corner_x = torch.unsqueeze(corner_x, dim=0)
        corner_y = torch.unsqueeze(corner_y, dim=0)
        corners = torch.cat((corner_x, corner_y), dim=0)
        # corners are top-left corners for each cell in the grid
        corners = corners.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        pred_box = output.clone()

        # for each bounding box
        for i in range(self.num_boxes):
            # x and y
            pred_box[:, i*5, :, :] = corners[:, 0, :, :] + output[:, i*5, :, :] * self.grid_size
            pred_box[:, i*5+1, :, :] = corners[:, 1, :, :] + output[:, i*5+1, :, :] * self.grid_size
            # w and h
            pred_box[:, i*5+2, :, :] = output[:, i*5+2, :, :] * self.image_size
            pred_box[:, i*5+3, :, :] = output[:, i*5+3, :, :] * self.image_size

        return pred_box


    # forward pass of the YOLO network
    def forward(self, x):
        # raw output from the network
        output = self.network(x).reshape((-1, self.num_boxes * 5 + self.num_classes, 7, 7))
        # compute bounding boxes in the original image space
        pred_box = self.transform_predictions(output)
        return output, pred_box


# run this main function for testing
if __name__ == '__main__':
    network = YOLO(num_boxes=2, num_classes=1)
    print(network)

    image = np.random.uniform(-0.5, 0.5, size=(1, 3, 448, 448)).astype(np.float32)
    image_tensor = torch.from_numpy(image)
    print('input image:', image_tensor.shape)

    output, pred_box = network(image_tensor)
    print('network output:', output.shape, pred_box.shape)

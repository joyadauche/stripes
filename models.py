## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output size = (W-F)/S +1 = (224-5)/1 +1 = 220 => (32, 220, 220)
        self.pool1 = nn.MaxPool2d(2, 2) # after one pool layer, this becomes (32, 110, 110)
        
        self.conv2 = nn.Conv2d(32, 64, 5) # output size = (W-F)/S +1 = (110-5)/1 +1 = 106 => (64, 106, 106)
        self.pool2 = nn.MaxPool2d(2, 2) # after two pool layer, this becomes (64, 53, 53)
        
        self.conv3 = nn.Conv2d(64, 96, 5) # output size = (W-F)/S +1 = (53-5)/1 +1 = 49 => (96, 49, 49)
        self.pool3 = nn.MaxPool2d(2, 2) # after three pool layer, this becomes (96, 24, 24)
        self.batch_norm = nn.BatchNorm2d(96)
        
        self.conv4 = nn.Conv2d(96, 128, 5) # output size = (W-F)/S +1 = (24-5)/1 +1 = 20 => (128, 20, 20)
        self.pool4 = nn.MaxPool2d(2, 2) # after three pool layer, this becomes (96, 10, 10)
        
        self.conv5 = nn.Conv2d(128, 164, 3) # output size = (W-F)/S +1 = (10-3)/1 +1 = 8 => (164, 8, 8)
        
        self.fc1 = nn.Linear(164*8*8, 50)
        
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(50, 136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.batch_norm(self.pool3(F.relu(self.conv3(x))))
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
    
class Net_two(nn.Module):
    def __init__(self):
        super(Net_two, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(4, 4), stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=136)
        
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout6 = nn.Dropout(p=0.6)
        
        self.bn1 = nn.BatchNorm2d(num_features=96, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=384, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn5 = nn.BatchNorm1d(num_features=4096, eps=1e-05)
        
        print('Modules: ', self.modules)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = I.xavier_uniform(m.weight, gain=1)
            elif isinstance(m, nn.Linear):
                m.weight = I.xavier_uniform(m.weight, gain=1)
       
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout4(x)
        
        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.dropout4(x)
        
        x = F.relu(self.conv5(x))
        x = self.bn4(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) # reshape -> Flattening
        
        x = F.relu(self.fc1(x))
        x = self.bn5(x)
        x = self.dropout6(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn5(x)
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x
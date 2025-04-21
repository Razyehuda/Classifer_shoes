import numpy as np
import torch
import torch.nn as nn
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 208634287

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000
K = 3
kernel = 3
drop_out = 0.2
class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = K
        self.n = n
        kernel_size = kernel
        padding = (kernel_size -1)//2

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()

        self.norm2d1=nn.BatchNorm2d(num_features=self.n)
        self.norm2d2=nn.BatchNorm2d(num_features=self.n*2)
        self.norm2d3=nn.BatchNorm2d(num_features=self.n*4)
        self.norm2d4=nn.BatchNorm2d(num_features=self.n*8
        )
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(drop)  # Adjust the dropout rate as needed
        self.dropout = nn.Dropout(drop_out)
        self.norm1d=nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(8*n * 14 * 28, 100)
        self.fc2 = nn.Linear(100, 2)
    
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
    
        inp = self.norm2d1(self.maxpool(self.relu(self.conv1(inp))))

        inp = self.norm2d2(self.maxpool(self.relu(self.conv2(inp))))

        inp = self.norm2d3(self.maxpool(self.relu(self.conv3(inp))))

        inp = self.norm2d4(self.maxpool(self.relu(self.conv4(inp))))

        
        inp = inp.reshape(-1, 8*self.n*14*28)
        inp = self.norm1d(self.relu(self.fc1(inp)))
        out = self.fc2(inp)
        out = self.sigmoid(out)
        # TODO: complete this function
        #out = torch.max(torch.relu(out),1)
        #out = torch.matrix_powerx
        #out= torch.Tensor(out)
        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        n=2*K
        self.n = n
        kernel_size =kernel
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=self.n, kernel_size=kernel_size, padding=padding, stride = 2)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2*self.n, kernel_size=kernel_size, padding=padding, stride = 2)
        self.conv3 = nn.Conv2d(in_channels=2*self.n, out_channels=4*self.n, kernel_size=kernel_size, padding=padding, stride = 2)
        self.conv4 = nn.Conv2d(in_channels=4*self.n, out_channels=8*self.n, kernel_size=kernel_size, padding=padding, stride = 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()
        self.norm2d1=nn.BatchNorm2d(num_features=self.n)
        self.norm2d2=nn.BatchNorm2d(num_features=self.n*2)
        self.norm2d3=nn.BatchNorm2d(num_features=self.n*4)
        self.norm2d4=nn.BatchNorm2d(num_features=self.n*8)
        self.norm1d=nn.BatchNorm1d(num_features=100)
        
        self.fc1 = nn.Linear(8 * self.n * 14 * 14, 100)
        self.dropout = nn.Dropout(drop_out)  # Adjust the dropout rate as needed
        self.fc2 = nn.Linear(100, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size

            2 := same/different pair
        '''
        inp = torch.cat((inp[:, :, :224, :], inp[:, :, 224:, :]), axis=1)

        inp =  self.norm2d1((self.relu(self.conv1(inp))))
        inp =  self.norm2d2((self.relu(self.conv2(inp))))
        inp =  self.norm2d3((self.relu(self.conv3(inp))))
        inp =  self.norm2d4((self.relu(self.conv4(inp))))
        inp = inp.reshape(-1, 8*self.n*14*14)
        inp = self.norm1d(self.relu(self.fc1(inp)))
        out = self.fc2(inp)
        out= self.sigmoid(out)
        return out
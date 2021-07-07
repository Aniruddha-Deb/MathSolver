import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from PIL import Image, ImageOps

class NeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2420, 100)
        self.fc2 = nn.Linear(100, 14)
        self.transform = transforms.ToTensor()


        self.load_state_dict(torch.load('mathsolver/value_net_dict', map_location=self.device))
        self.eval()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x

    def solve(self, img: Image):
        img_width, img_height = img.size
        op_str = '0123456789+-*/'
        leftimg = self.transform(img.crop((0,0,img_height,img_height))).float()
        midimg = self.transform(img.crop((img_height,0,img_height*2,img_height))).float()
        rightimg = self.transform(img.crop((img_height*2,0,img_height*3,img_height))).float()

        sections = torch.stack([leftimg, midimg, rightimg])
        sections = sections.to(self.device)
        output = self(sections)
        predictions = torch.argmax(output, dim=1)

        print(predictions)

        l, m, r = predictions
        if l >= 10: # prefix
            return int(eval(f'{m}{op_str[l]}{r}'))
        elif m >= 10: # infix
            return int(eval(f'{l}{op_str[m]}{r}'))
        elif r >= 10: # postfix
            return int(eval(f'{l}{op_str[r]}{m}'))
        else:
            return 0

network = NeuralNetwork()

# testing
#print(network.solve(ImageOps.grayscale(Image.open('../out.png'))))
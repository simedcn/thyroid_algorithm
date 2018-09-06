import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

#torch.manual_seed(1)    # reproducible
import numpy as np
import os 
BATCH_SIZE = 64      
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(684)
        self.fc1 = nn.Linear(684, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 4)
        #self.bn3 = nn.BatchNorm1d(256)
        #self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(self.bn1(x)))
        #x = F.relu(self.fc2(self.bn2(x)))
        x = F.log_softmax(self.fc2(self.bn2(x)))
        return x

def Get_Data(path,tag=""):
    if tag!= "":
        print("Loading "+tag+" ...")
    ClassList = ["Bact","Fungal","Others","Virus"]
    cls_num = 0
    data = []
    target =[]
    for c_i in ClassList:
        print(c_i)
        count = 0
        c_path = os.path.join(path,c_i)
        FileList = os.listdir(c_path)
        for f_i in FileList:
            f_path = os.path.join(c_path,f_i)
            x = np.loadtxt(f_path)
            y = cls_num
            data.append(x)
            target.append(y)
            count+=1
            if count%1000==0:
                print(count)
        cls_num+=1
    data = np.array(data) 
    target = np.array(target)
    data_t = torch.from_numpy(data).float()
    target_t = torch.from_numpy(target).long()
    print(type(data_t))
    torch_dataset = Data.TensorDataset(data_t, target_t)
    return torch_dataset

def train(epoch, net, trainLoader, optimizer):
    net.train()
    tot_loss = 0
    tot_err = 0
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*float(incorrect)/len(data)
        tot_loss+=loss.data[0]
        tot_err+=err
    tot_loss=tot_loss/(batch_idx+1)
    tot_err=tot_err/(batch_idx+1)
    print('Train Epoch: {:.2f} \t Loss: {:.6f}\tError: {:.6f}'.format(epoch,tot_loss,tot_err))
    
def test(net, testLoader):
    net.eval()
    tot_loss = 0
    tot_err = 0
    tot = 0
    count_right = [0]*4
    count_tot = [0]*4
    matrix = [[0] * 4 for row in range(4)]
    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = net(data)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1)[1]
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*float(incorrect)/len(data)
        tot_loss+=loss.data[0]
        tot_err+=err
        for i in range(target.cpu().data.size()[0]):
            #print(output.cpu().data[i])
            curr_max = torch.max(output.cpu().data[i])
            curr_min = torch.min(output.cpu().data[i])
            index = target.cpu().data[i]#actually right
            count_tot[index]+=1             

            for j in range(4):
                if output.cpu().data[i][j] == curr_max:
                    pred = j
            matrix[index][pred]+=1
            if pred == index:
                count_right[index]+=1 
                tot+=1
    tot_loss=tot_loss/(batch_idx+1)
    tot_err=tot_err/(batch_idx+1)
    print('Test Epoch: {:.2f} \t Loss: {:.6f}\tError: {:.6f}'.format(epoch,tot_loss,tot_err))
    print(count_right)
    print(count_tot)
    print(matrix)

train_dataset = Get_Data("./feature_v2/train/","train")
test_dataset = Get_Data("./feature_v2/test/","test")

trainLoader = Data.DataLoader(
    dataset=train_dataset,      
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    #num_workers=5,              
)

testLoader = Data.DataLoader(
    dataset=test_dataset,      
    batch_size=BATCH_SIZE,      
    shuffle=False,               
    #num_workers=5,              
)

net = Net()
net = net.cuda()

opt = "adam"

if opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
elif opt == 'adam':
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
elif opt == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)


for epoch in range(10): 
    train(epoch, net, trainLoader, optimizer)
    test(net, testLoader)
       

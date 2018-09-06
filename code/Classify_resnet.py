import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import pretrainedmodels.utils as utils
import pretrainedmodels

#torch.manual_seed(1)    # reproducible
import numpy as np
import os 
BATCH_SIZE = 128      
number_class=2 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(100610)#1258
        self.fc1 = nn.Linear(100610, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.log_softmax(self.fc3(self.bn3(x)))
        return x


def Get_Data(path,tag=""):
    load_img = utils.LoadImage()
    model_name = 'fbresnet152'  # could be fbresnet152
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    tf_img = utils.TransformImage(model)
    if tag!= "":
        print("Loading "+tag+" ...")
    #ClassList = ["B","M"]
    cls_num = 0
    data = []
    target =[]
    txtpath="/home/shaoping/image_data/DDTI_data_traintestmix2/txt_selected/"

    for folder1 in os.listdir(path):
        print(folder1)
        count = 0
        for item in os.listdir(path + '/'+ folder1):
            path_img = path + '/' + folder1
            input_img = load_img(path_img + '/' + item)
            input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
            input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            input = torch.autograd.Variable(input_tensor, requires_grad=False)
            output_logits = model(input)
            output_features=model.features(input)
            x=output_features.view(1,-1)[0]
            print("type of x:",x)
            x1=output_logits[0]
            print("type of x1:",x1)
            #print("x: ", x)

            #txt feature:
            txtname=txtpath+item[:-3]+'txt'
            #print(txtname)
            f = open(txtname.split('#')[0]+'.txt', 'r')
            ratio = f.readline()
            insideline = f.readline()
            outsideline = f.readline()
            cs = f.readline()
            nums = ratio.split(',')
            #print(nums[0].split(':')[1], nums[1].split(':')[1],float(nums[0].split(':')[1]) / float(nums[1].split(':')[1]))
            rationum=float(nums[0].split(':')[1]) / float(nums[1].split(':')[1])
            #print(cs.split(',')[2].split(':')[1])
            csnum=cs.split(',')[2].split(':')[1]
            inside, outside = [], []
            list1 = insideline.split(':')[1][1:-2].split(', ')
            list2 = outsideline.split(':')[1][1:-2].split(', ')
            for i in list1:
                inside.append(int(i))
            for i in list2:
                outside.append(int(i))
            diff = []
            for i in range(len(list1)):
                diff.append(outside[i] - inside[i])
            #print(diff)
            f.close()
            x_np=x.detach().numpy()
            #print("x_np:",x_np)
            x_numpy=x_np.tolist()
            #print(x_numpy)
            x_numpy+=diff
            x_numpy.append(rationum)
            x_numpy.append(float(csnum[:-2]))
            #print(float(csnum[:-2]))
            #print(x_numpy)
            x_numpy=np.array(x_numpy)

            #print(type(x_numpy),np.size(x_numpy))
            #print("x_numpy: ",x_numpy)
            y = cls_num
            data.append(x_numpy)
            target.append(y)
            print(item)
            count += 1
            if count % 1000 == 0:
                print(count)
        cls_num += 1
    #print("before array: ", data)
    data = np.array(data)
    #print("after array: ",type(data))
    #print(data)
    target = np.array(target)
    data_t = torch.from_numpy(data).float()
    target_t = torch.from_numpy(target).long()
    #print("data_t: ",data_t.size())
    torch_dataset = Data.TensorDataset(data_t, target_t)
    return torch_dataset

""" 
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
"""

def train(epoch, net, trainLoader, optimizer):
    net.train()
    tot_loss = 0
    tot_err = 0
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        #print(data.size(),type(data))
        #print(data)
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
    count_right = [0]*number_class
    count_tot = [0]*number_class
    matrix = [[0] * number_class for row in range(number_class)]
    for batch_idx, (data, target) in enumerate(testLoader):
        #print("batch_idx and data and target is:",batch_idx,data,target)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = net(data)
        loss = F.nll_loss(output, target)
        print("target is:",target)
        pred = output.data.max(1)[1]
        print("pred is:",pred)
        incorrect = pred.ne(target.data).cpu().sum()
        print("incorrect is",incorrect)
        err = 100.*float(incorrect)/len(data)
        tot_loss+=loss.data[0]
        tot_err+=err
        for i in range(target.cpu().data.size()[0]):
            #print(output.cpu().data[i])
            curr_max = torch.max(output.cpu().data[i])
            curr_min = torch.min(output.cpu().data[i])
            index = target.cpu().data[i]#actually right
            count_tot[index]+=1             

            for j in range(number_class):
                if output.cpu().data[i][j] == curr_max:
                    pred = j
            matrix[index][pred]+=1
            #print("pred and index is:",pred,index)
            if pred == index:
                count_right[index]+=1 
                tot+=1
    tot_loss=tot_loss/(batch_idx+1)
    tot_err=tot_err/(batch_idx+1)
    print('Test Epoch: {:.2f} \t Loss: {:.6f}\tError: {:.6f}'.format(epoch,tot_loss,tot_err))
    print(count_right)
    print(count_tot)
    print(matrix)
    acc=(matrix[0][0]+matrix[1][1])/np.sum(matrix)
    print("accuracy is: ",acc)

"""
train_dataset = Get_Data("./feature_v2/train/","train")
test_dataset = Get_Data("./feature_v2/test/","test")
"""
train_dataset = Get_Data("/home/shaoping/image_data/DDTI_data_traintestmix2/train/","train")
test_dataset = Get_Data("/home/shaoping/image_data/DDTI_data_traintestmix2/test/","test")


trainLoader = Data.DataLoader(
    dataset=train_dataset,      
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    #num_workers=5,              
)
#print(trainLoader.size(),type(trainLoader))
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


for epoch in range(5): 
    train(epoch, net, trainLoader, optimizer)
    test(net, testLoader) 


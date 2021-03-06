import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import pretrainedmodels.utils as utils
import pretrainedmodels

#torch.manual_seed(1)    # reproducible
import numpy as np
import os
import json
BATCH_SIZE = 64
number_class=2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(401872)#206+1258
        self.fc1 = nn.Linear(401872, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 256)
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
    maxcount=206 #(xpoints,ypoints)

    for folder1 in os.listdir(path):
        print(folder1)
        count = 0
        for item in os.listdir(path + '/'+ folder1):
            # for open json文件
            #cy 囊实性 position 结节位置 ratio纵横比
            with open("./record_test.json", "r") as file:
                json_dict = json.loads(json_str)
                cy=json_dict[0]
                position=json_dict[1]
                ratio=json_dict[2]
                file.close()

            #load tuber_img,normal_img，margin_img,shape_img
            path_img = path + '/' + folder1

            tuber_img = load_img(path_img + '/' + item)
            input_tensor1 = tf_img(tuber_img)  # 3x400x225 -> 3x299x299 size may differ
            input_tensor1 = input_tensor1.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            tuber_input = torch.autograd.Variable(input_tensor1, requires_grad=False)

            normal_img = load_img(path_img + '/' + item)
            input_tensor2 = tf_img(normal_img)  # 3x400x225 -> 3x299x299 size may differ
            input_tensor2 = input_tensor2.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            normal_input = torch.autograd.Variable(input_tensor2, requires_grad=False)

            margin_img = load_img(path_img + '/' + item)
            input_tensor3 = tf_img(margin_img)  # 3x400x225 -> 3x299x299 size may differ
            input_tensor3 = input_tensor3.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            margin_input = torch.autograd.Variable(input_tensor3, requires_grad=False)

            shape_img = load_img(path_img + '/' + item)
            input_tensor4 = tf_img(shape_img)  # 3x400x225 -> 3x299x299 size may differ
            input_tensor4 = input_tensor4.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            shape_input = torch.autograd.Variable(input_tensor4, requires_grad=False)

            #load models:
            margin_model = torch.load("margin_model.pkl")
            shape_model = torch.load("shape_model.pkl")
            echo_model = torch.load("echo_model.pkl")

            #extract feature
            tuber_features = model.features(tuber_input).view(1,-1)[0]
            margin_features=margin_model.feature(margin_input).view(1,-1)[0]
            shape_features=shape_model.feature(shape_input).view(1,-1)[0]
            echo_input=np.array(tuber_input)-np.array(normal_input)
            echo_features=echo_model.feature(echo_input).view(1,-1)[0]
            print("margin_features:",margin_features)
            print("margin_features:", margin_features)
            print("margin_features:", margin_features)
            print("margin_features:", margin_features)

            x_numpy=tuber_features+margin_features+shape_features+echo_features
            x_numpy.append(cy)
            x_numpy.append(position)
            x_numpy.append(ratio)

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


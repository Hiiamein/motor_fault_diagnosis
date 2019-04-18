import numpy as np
import torch 
import torchvision
import torch.nn as nn
import sklearn as sk
from sklearn import preprocessing


train_0_F = np.loadtxt(open("../data/motor_fault/train_0_F.csv","rb"), delimiter=",", skiprows=0)
train_1_F = np.loadtxt(open("../data/motor_fault/train_1_F.csv","rb"), delimiter=",", skiprows=0)
# 正向测试集
#test_F =  np.loadtxt(open("../data/motor_fault/test_F.csv","rb"), delimiter=",", skiprows=0)
print("train_0_F.shape: ", train_0_F.shape)
print("train_1_F.shape: ", train_1_F.shape)
#print("test_F.shape",test_F.shape)


# 高斯归一化
train_0_F[:,:-1] = preprocessing.scale(train_0_F[:,:-1])
train_1_F[:,:-1] = preprocessing.scale(train_1_F[:,:-1])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 14
hidden_size1 = 100
hidden_size2 = 60
hidden_size3 = 30
num_classes = 2
num_epochs = 30
t_max = 10
batch_size = 100
learning_rate = 1e-2
valid_size = 0.2

# Dataset
train_F = np.concatenate((train_0_F, train_1_F), axis=0)
x = torch.tensor(train_F[:,:-1], dtype=torch.float32)
y = torch.tensor(train_F[:,-1], dtype=torch.long)
print("x.size: {}".format(x.size()))
print("y.size: {}".format(y.size()))
dataset = torch.utils.data.TensorDataset(x, y)

num_train = len(dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]
trainset = torch.utils.data.Subset(dataset=dataset, indices=train_idx)
validset = torch.utils.data.Subset(dataset=dataset, indices=valid_idx)

trainloader = torch.utils.data.DataLoader(trainset,
                                     batch_size = batch_size,
                                     shuffle = True,
                                     num_workers = 2)
validloader = torch.utils.data.DataLoader(validset,
                                     batch_size = batch_size,
                                     shuffle = False,
                                     num_workers = 2)


# Define Model - Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=input_size, hidden_size1=hidden_size1, 
                 hidden_size2=hidden_size2, hidden_size3=hidden_size3, num_classes=num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


model = NeuralNet().to(device)
weight=torch.tensor([30,500]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = t_max)


def test(loader):
    # test on all test data
    correct = 0 
    total = 0
    total0 = 0
    total1 = 0
    p00 = 0 #第一个0代表label，第二个0代表预测值
    p01 = 0
    p10 = 0
    p11 = 0    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            predicted = torch.max(outputs.data, 1)[1]
            total += y.size(0)
            total1 += sum(y.cpu().numpy())
            correct += (predicted == y).sum().item()
            for i in range(batch_size):
                if y[i] == 0 and predicted[i] == 0:
                    p00 += 1
                elif y[i] == 0 and predicted[i] == 1:
                    p01 += 1
                elif y[i] == 1 and predicted[i] == 0:
                    p10 += 1
                elif y[i] == 1 and predicted[i] == 1:
                    p11 += 1
                
    total0 = total - total1
    my_acc = 100 * correct / total
    my_recall = p11 / total1
    my_precision = p00 / total0 #大赛的要求，和课本上的精度定义不一样
    
    model.train()
    return my_acc, my_recall, my_precision


# Train the model 
total_step = len(trainloader)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (x, y) in enumerate(trainloader):
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device)
  
        # Forward pass
        outputs = model(x)
        #print(outputs)
        loss = criterion(outputs,y)
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        if (i+1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, running_loss/100))
            running_loss = 0.0
        lr_scheduler.step()
    
    # save model
    if epoch > 14:
        torch.save(model, 'checkpoint_F_epoch_{}.pt'.format(epoch+1))
    train_acc, train_recall, train_precision = test(trainloader)
    print("train_acc: {:.4f}, train_recall: {:.4f}, train_precision: {:.4f}".format(train_acc, train_recall, train_precision))
    valid_acc, valid_recall, valid_precision = test(validloader)
    print("train_acc: {:.4f}, train_recall: {:.4f}, train_precision: {:.4f}".format(test_acc, test_recall, test_precision))







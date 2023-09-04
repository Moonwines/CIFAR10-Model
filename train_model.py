import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dkf_model import*

train_data = torchvision.datasets.CIFAR10("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)


train_data_size = len(train_data)
test_data_size = len(test_data)
print(test_data_size)
print(train_data_size)

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#创建神经网络模型
dkf = Dkf()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(dkf.parameters(),lr=learning_rate)

#设置训练网络的参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10
#添加tensorboard
writer = SummaryWriter("logs_train")
#添加准确率
total_accuracy = 0
for i in range(epoch):
    print("----------第{}轮训练开始------------".format(i))
    for data in train_dataloader:
        imgs,targets = data
        outputs = dkf(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        total_train_step +=1
        if total_train_step%100==0:
            print("训练次数：{},Loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = dkf(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step+=1

    torch.save(dkf,"dkf_{}.pth".format(i))
    print("模型已保存")



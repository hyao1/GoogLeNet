import os, sys
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as opt
import model
from torch.optim import lr_scheduler
import pc_loss
import time

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)

lr = 0.01
datasets_path = "/content/drive/My Drive/CrossX-master/utils/cubbirds/"
pymodel = "/content/drive/My Drive/CrossX-master/pymodels"
batch_size = 16
img_size = 448
epochs = 30
dataset_name = 'cubbirds'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 数据预处理
data_transform = {
    "trainval": transforms.Compose(
        [transforms.Resize((600, 600)),
         transforms.RandomCrop((img_size, img_size)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation((0, 30), center=(224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),

    "test": transforms.Compose(
        [transforms.Resize((600, 600)),
         transforms.CenterCrop((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
}
train_dataset = datasets.ImageFolder(root=datasets_path + "trainval", transform=data_transform["trainval"])
validate_dataset = datasets.ImageFolder(root=datasets_path + "test", transform=data_transform["test"])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8)

print('模型调用')

# 定义损失函数
loss_function = nn.CrossEntropyLoss()
pc_lossfuntion = pc_loss.PCLoss()

train_len = len(train_loader.dataset)
all_train_step = int(train_len / batch_size)

test_len = len(validate_loader.dataset)
all_test_step = int(test_len / batch_size)

print('开始训练')
for i in range(2):
    if i == 0:
        print("全局平均池化代替全连接")
        googlenet = model.googlenet(num_classes=200, pretrained=True, aux_logits=True, fc_to_avg=True)
    else:
        print("不使用全局平均池化")
        googlenet = model.googlenet(num_classes=200, pretrained=True, aux_logits=True)

    if torch.cuda.device_count() > 0:
        googlenet = nn.DataParallel(googlenet)
    googlenet.to(device)

    # 定义优化器
    optmeth = 'sgd'
    optimizer = opt.SGD(googlenet.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # 训练阶段
        scheduler.step()
        googlenet.train()
        running_loss = 0.0
        running_acc = 0.0
        since = time.time()
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits, aux2_logits, aux1_logits = googlenet(inputs)
            _, predict = torch.max(logits, dim=1)

            loss_logits = loss_function(logits, labels)
            loss_aux2_logits = loss_function(aux2_logits, labels)
            loss_aux1_logits = loss_function(aux1_logits, labels)
            loss = loss_logits + loss_aux1_logits * 0.3 + loss_aux2_logits * 0.3

            # 加入pc混淆
            pcloss = pc_lossfuntion(logits)
            loss = loss + 10 * pcloss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += torch.sum(predict == labels.data).item()

            print("\r{}\{} train loss: {:.3f} train acc: {:.3f}".format(step, all_train_step, loss,
                                                                        running_acc / train_len), end="")
        time_elapsed = time.time() - since
        print('this epoch complete in {:.0f}min {:.0f}sec'.format(time_elapsed // 60, time_elapsed % 60))

        # 测试阶段
        googlenet.eval()
        running_loss = 0.0
        running_acc = 0.0
        for step, data in enumerate(validate_loader, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _, _ = googlenet(inputs)
            _, predict = torch.max(logits, dim=1)

            loss = loss_function(logits, labels)

            running_loss += loss.item()
            running_acc += torch.sum(predict == labels.data).item()
            print("\r正在计算test_loss,test_acc:{}\{}".format(step, all_test_step), end="")
        print("\rtest loss: {:.3f} test acc: {:.3f}".format(running_loss / test_len, running_acc / test_len), end="")
        print()

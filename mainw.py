'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os,csv
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochspan', default=200, type=int, help='learning rate')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--nr', default=0.2, type=float, help='set noise level')
# "fix", "decrease", "step"
parser.add_argument('--nr_mode', default="fix", help='set noise level')
#zs or pj
parser.add_argument('--mode', default="zs", help='zhurui or pingjun')
#model: "resnet50"; "googlenet";"densenet121"
parser.add_argument('--model', default="resnet50", help='set model type')
#0: batch 1: layer
parser.add_argument('--dim', default=0, type=int, help='batch or layer')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=6)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.model == "resnet50":
    if args.mode == "zs":
        net = ResNetW50(True,args.dim)
    else:
        net = ResNetW50(False,args.dim)

if args.model == "googlenet":
    if args.mode == "zs":
        net = GoogLeNetW(True,args.dim)
    else:
        net = GoogLeNetW(False,args.dim)

if args.model == "densenet121":
    if args.mode == "zs":
        net = DenseNetW121(True,args.dim)
    else:
        net = DenseNetW121(False,args.dim)

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if False:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load("./checkpoint/%s_%s_%f_%d.t7"  % (args.model, args.mode, args.nr,args.dim))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120])

def Adjustment_mode(epoch):
    if args.nr_mode == "fix":
        val = args.nr

    if args.nr_mode == "decrease":
        p = 2 * args.epochspan //3
        val = args.nr * (p-epoch)/p
        if val<0:
            val = 0

    if args.nr_mode == "step":    
        if epoch<  args.epochspan //3:
            val = args.nr
        elif epoch< 2 * args.epochspan //3:
            val = args.nr * 0.5
        else:
            val = 0

    return val

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    scheduler.step()

    val = Adjustment_mode(epoch)
    net.zhushui.set(val)
    net.pingjun.set(val)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f'
            % (train_loss/(batch_idx+1), 100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    predlist = []
    targlist = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            predlist += predicted.tolist()
            targlist += targets.tolist()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f'
                % (test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    f1 = f1_score(targlist, predlist, average='macro' )
    p = precision_score(targlist, predlist, average='macro')
    r = recall_score(targlist, predlist, average='macro')

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        fname = "./checkpoint/%s_%s_%f_%d.t7"  % (args.model, args.mode, args.nr,args.dim)
        torch.save(state, fname)
        best_acc = acc
    return acc, test_loss/len(testloader), p,r,f1

Loss_list = []
Accuracy_list = []
plist = []
rlist = []
f1list =[]

for epoch in range(args.epochspan):
    train(epoch)
    acc,loss,p,r,f1 = test(epoch)
    Loss_list.append(loss)
    Accuracy_list.append(acc)
    plist.append(p)
    rlist.append(r)
    f1list.append(f1)


Loss_list.append(min(Loss_list))
Accuracy_list.append(max(Accuracy_list))
x1 = range(args.epochspan)
x2 = range(args.epochspan)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1[:-1], 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2[:-1], '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
#plt.show()
figname = "./checkpoint/accyracyloss_%s_%s_%f_%d.jpg" % (args.model, args.mode, args.nr, args.dim)
plt.savefig(figname)
csvfile = "./checkpoint/result_%s_%s_%f_%d.csv" % (args.model, args.mode, args.nr, args.dim)
out = open(csvfile, "w", newline = "")
csv_writer = csv.writer(out, dialect = "excel")
csv_writer.writerow(Accuracy_list)
csv_writer.writerow(Loss_list)
csv_writer.writerow(plist)
csv_writer.writerow(rlist)
csv_writer.writerow(f1list)

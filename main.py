import torch
import torchvision
import torch.nn as nn
import model
import torch.optim as optim
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def train(n_epochs, optimizer, model, lossFn, trainLoader, batch_size = 128):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], last_epoch=-1)

    for epoch in range(1,n_epochs+1):
        lossTrain = 0
        for imgs, labels in trainLoader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs).to(device=device)

            loss = lossFn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lossTrain += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print(' Epoch {}, Training loss {}'.format(epoch,lossTrain / len(trainLoader)))

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32,4)
    ,torchvision.transforms.ToTensor()
    ,torchvision.transforms.Normalize(mean=[0.50707516, 0.48654887, 0.44091784], std = [0.26733429, 0.25643846, 0.27615047])
])


train_set = torchvision.datasets.CIFAR100(root = './train_set', download=True, train = True,transform=transform)
test_set = torchvision.datasets.CIFAR100(root = './test_set', download=True, train = False,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.50707516, 0.48654887, 0.44091784], std = [0.26733429, 0.25643846, 0.27615047])]))


train_loader = torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

model = model.EfficientNet("b0", 100).to(device=device)

learningRate = 0.1
optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum=0.9, weight_decay=1e-4)
lossFn = nn.CrossEntropyLoss(ignore_index=-1)

train(200, optimizer, model, lossFn, train_loader)

total = 0
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        outputs = model(imgs)
        _, pred = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (pred==labels).sum().item()
    print('test Accuracy : {}'.format(100 * correct / total))








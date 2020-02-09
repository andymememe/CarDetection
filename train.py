from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models

import numpy as np
import os

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def train_one_epoch(x, y, epoch, batch, model, criterion, optimizer, device):
    model.train()
    
    nBatch = (len(x) // batch) + 1
    for bID in range(nBatch):
        xBatch = [xi.unsqueeze(0).to(device) for xi in x[bID * batch: (bID + 1) * batch]]
        yBatch = [torch.tensor([yi]).to(device) for yi in y[bID * batch: (bID + 1) * batch]]
        xBatch = torch.cat(xBatch, dim=0)
        yBatch = torch.cat(yBatch, dim=0)

        opt = model(xBatch)
        loss = criterion(opt, yBatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (bID + 1) % 100 == 0:
            print(f"Epoch {epoch} Batch {bID + 1}: Loss: {loss.item()}")

def evaluate(x, y, epoch, model, criterion, device):
    model.eval()
    losses = []
    acc1s = []
    acc5s = []
    with torch.no_grad():
        for image, target in zip(x, y):
            image = image.unsqueeze(0).to(device, non_blocking=True)
            target = torch.tensor([target]).to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())
    lossAvg = sum(losses) / len(losses)
    acc1Avg = sum(acc1s) / len(acc1s)
    acc5Avg = sum(acc5s) / len(acc5s)
    print(f"Epoch {epoch}: Loss: {lossAvg}, Acc. top 1: {acc1Avg}, Acc. top 5: {acc5Avg}")

    return lossAvg

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

numberClass = 196
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
model = models.resnet50(pretrained=True, progress=False)
oriInFeature = model.fc.in_features
model.fc = nn.Linear(oriInFeature, numberClass)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
criterion = nn.CrossEntropyLoss()

print('Loading data...')
trainX = []
trainY = []
trainFile = []
datasetRoot = "../../Reference/Dataset/Car"
with open(os.path.join(datasetRoot, "meta", "cars_train_annos.txt"), 'r') as f:
    for line in f:
        data = line.strip().split('\t')

        classType = int(data[4]) - 1
        trainY.append(classType)

        trainFile.append(data[-1])
        img = Image.open(os.path.join(datasetRoot, "train", data[-1]))
        trainX.append(transform(img.convert("RGB")))

print('Total train...')
epoch = 10
batch = 32
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)
lrScheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

for e in range(epoch):
    train_one_epoch(
        trainX, trainY,
        e, batch,
        model, criterion, optimizer, device
    )
    lrScheduler.step()
    evaluate(trainX, trainY, e, model, criterion, device)
torch.save(model.state_dict(), 'model/car_detection.model')
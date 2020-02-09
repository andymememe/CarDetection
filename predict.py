from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models

import os
import shutil

datasetRoot = "../../Reference/Dataset/Car"

def evaluate(x, files, classes, model, device):
    model.eval()
    with torch.no_grad():
        with open('result/test_result.txt', 'w') as f:
            for img, aFile in zip(x, files):
                img = img.unsqueeze(0).to(device)
                output = model(img)
                res = torch.argmax(output, dim=1).to(torch.device("cpu")).numpy().tolist()[0]
                f.write(f"{res}\n")
                modelName = classes[res]

                if not os.path.exists(os.path.join('result', modelName)):
                    os.makedirs(os.path.join('result', modelName))
                
                shutil.copy(
                    os.path.join(datasetRoot, "test", aFile)
                    os.path.join('result', modelName, aFile)
                )
        

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
model.load_state_dict(torch.load('model/car_detection.model'))
model.to(device)

print('Loading data...')
classes = []
with open(os.path.join(datasetRoot, "meta", "cars_meta.txt"), 'r') as f:
    for line in f:
        classes.append(line.strip())

testX = []
testFiles = []
for aFile in os.listdir(os.path.join(datasetRoot, "test")):
    testFiles.append(aFile)
    img = Image.open(os.path.join(datasetRoot, "test", aFile))
    testX.append(transform(img.convert("RGB")))

print('Testing...')
evaluate(testX, testFiles, classes, model, device)
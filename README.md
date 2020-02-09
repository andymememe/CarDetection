# CarDetection
Car Model detection models finetuned from a pretrained ResNet50 model. Using Pytorch.

## Dataset
[Cars Dataset from Standford AI Lab](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* You have to download the dataset from link above and put it the location you like. Also you have to change the path of datasetRoot in [train.py](train.py) and [predict.py](predict.py)
* I parsed the meta file that originally provided with Matlab matrix and save to txt file in [dataset/meta](meta) folder.
* It has 196 classes of car model.

## Training
* I finetuned the pretrained model with a new FC layer for the last layer of the model.
* 10 Epoch, batch number is 32
* I used a RTX2080 GPU (8 GB)

## Result
* Testing with official test set.
* Accuracy: 80.79 %

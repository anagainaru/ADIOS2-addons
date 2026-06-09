# Intructions to run the training demo on Gray-Scott data

Used for the ISC tutorial 2026.

Steps:
- Train a visual transformer that predicts if an image is good or bad having an input the actual image
- Create a campaign with runs of Gray-Scott for different input parameters (Du, Dv, F, k)
- Next


1. Visual transformer

The tutorial uses a hybrid model that combines the Vision Transformer (ViT) design with a pre-trained ResNet18.

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
```

Usage:
```
python train_image_model.py --help
usage: train_image_model.py [-h] [--save SAVE] [--batches BATCHES] [--epochs EPOCHS] folder

positional arguments:
  folder             Folder containing one folder for each class of images to train on

options:
  -h, --help         show this help message and exit
  --save SAVE        path where to save image_model.pth
  --batches BATCHES

$ python train_image_model.py imgs --save models/image_model.pth
Epoch 1/10, loss=2.4178, val_acc=0.8571
Epoch 2/10, loss=0.6009, val_acc=0.8889
Epoch 3/10, loss=0.1521, val_acc=0.9048
Epoch 4/10, loss=0.0666, val_acc=0.9206
Epoch 5/10, loss=0.0280, val_acc=0.9048
Epoch 6/10, loss=0.0179, val_acc=0.9206
Epoch 7/10, loss=0.0122, val_acc=0.9048
Epoch 8/10, loss=0.0210, val_acc=0.9048
Epoch 9/10, loss=0.0089, val_acc=0.9048
Epoch 10/10, loss=0.0092, val_acc=0.9048
Saved model to models/image_model.pth
Classes: {'bad': 0, 'good': 1}
```

Predict the label of 10 random images from a given folder:
```
$ python predict_image_model.py --help
usage: predict_image_model.py [-h] [--model MODEL] [--num NUM] folder

positional arguments:
  folder         Folder containing images to predict

options:
  -h, --help     show this help message and exit
  --model MODEL
  --num NUM

$ python predict_image_model.py bigspace/imgs --model models/image_model.pth --num 4
Du0.122_Dv0.294_F0.02664_k0.0433_U010000_yz.png: bad (0.999)
Du0.152_Dv0.113_F0.02023_k0.0745_U010000_yz.png: bad (0.999)
Du0.188_Dv0.316_F0.01717_k0.0751_U010000_yz.png: bad (0.999)
Du0.332_Dv0.0812_F0.03782_k0.0622_U010000_yz.png: good (1.000)
```

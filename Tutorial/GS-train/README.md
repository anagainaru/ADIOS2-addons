# Intructions to run the training demo on Gray-Scott data

Used for the ISC tutorial 2026.

Steps:
- Train a visual transformer that predicts if an image is good or bad having an input the actual image
- Create a campaign with runs of Gray-Scott for different input parameters (Du, Dv, F, k)
- Next

```
$ cat requirements.txt
torch
torchvision
pillow
numpy
hpc-campaign
adios2
python-dateutil

$ ~/_penv/ai-adios-penv/bin/pip install -r requirements.txt

$ du -h -d 1 ~/_penv/ai-adios-penv/
699M	~/_penv/ai-adios-penv/

$ source ~/_penv/ai-adios-penv/bin/activate
```

## 1. Visual transformer for labeling images

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

## 2. Create a campaign with Gray-Scott runs

The `hpc-campaign` package should be installed in the python virtual environment.
The default path to the campaigns should be defined in `~/.config/hpc-campaign/config.yaml`

Read the docs in: [https://hpc-campaign.readthedocs.io](https://hpc-campaign.readthedocs.io)

The `create_campaign_archive.py` creates a campaign of all the runs in the `data` folder. Each folder in data has the format `Du[val]_Dv[val]_F[val]_k[val]` containing a gs.bp file with data and png files with images for U and V.

```
$ ls data
Du0.01_Dv0.212_F0.03519_k0.0517     Du0.0901_Dv0.0882_F0.01758_k0.0718  Du0.189_Dv0.378_F0.03883_k0.0797    Du0.287_Dv0.0804_F0.01046_k0.0496
Du0.0106_Dv0.347_F0.03968_k0.0756   Du0.0904_Dv0.337_F0.02749_k0.05     Du0.19_Dv0.191_F0.03627_k0.0726     Du0.287_Dv0.134_F0.02118_k0.0508
Du0.0112_Dv0.378_F0.0371_k0.0536    Du0.0913_Dv0.0397_F0.01447_k0.0707  Du0.191_Dv0.268_F0.03364_k0.0632    Du0.287_Dv0.264_F0.003423_k0.0742

$ ls data/Du0.01_Dv0.212_F0.03519_k0.0517
gs.bp         success.txt   U00400_yz.png U00800_yz.png V00200_yz.png V00600_yz.png V01000_yz.png
settings.json U00200_yz.png U00600_yz.png U01000_yz.png V00400_yz.png V00800_yz.png

$ python create_campaign_archive.py
```
The `create_campaign_archive.py` will create a file called `bigspace.aca`.

```
$ hpc_campaign manager --campaign_store . bigspace info
==========   info   ==================================================
ADIOS Campaign Archive, version 0.7, created on Jun  9 11:09

Hosts and directories:
  AnaLaptop02   longhostname = mac146553
     1. /Users/95j/_projects/adios/ISC-tutorial/bigspace

Other Datasets:
    cbac78222617300fb2563b85c997511b   ADIOS   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs
    3739477102ed3f18bcb7c56f3f2e4d9a   TEXT    Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/settings.json
    5628ffa333dd3cae87260b46a08af9d9   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/V/yz/00400
    f14015135f2836f0889cd6a3a2d8a4ca   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/U/yz/00800
    3a2e2b11a082315e8c1e06a0afde19a1   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/U/yz/00200
    9a7fb55852b53398aab3f19975e2f02a   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/V/yz/00200
    696fb92055ec3a0eafcc0ced7de8bf8b   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/V/yz/00800
    fb62b4ef196330e7996b13ca14736348   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/U/yz/00400
    1264392c2ce43d519f1b6d5561270a70   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/V/yz/00600
    35f4d9565bd33a47bbcee4e5c9f3ca45   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/V/yz/01000
    5a54e15db32a30deb7fd3a4313725ca5   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/U/yz/01000
    ad8f24e45a443c8fb906021206bda895   IMAGE   Jun  8 18:48   Du0.0595_Dv0.369_F0.03723_k0.0682/gs/U/yz/00600
```

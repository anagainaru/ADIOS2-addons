# Intructions to run the training demo on Gray-Scott data

_Used for the ISC tutorial 2026._

Steps:
1. On the small image dataset:
    - Train a visual transformer that predicts if an image is good or bad having an input the actual image
    - Train a XGBClassifier decision trees that takes the parameters and predicts if an image is good or not
2. Create campaigns:
    - Create a campaign with the images
    - Create a separate campaign with ADIOS runs of Gray-Scott for different input parameters (Du, Dv, F, k)
3. Training loop for the large dataset of runs
    - Code that chooses the next bach to train on, reads the GS data and generates images for step 1000
    - Create lables for all datasets by using the image model
    - Train or update the weights of the pre-trained XGBClassifier model (using the parameters and the labels)
4. Predict using the XGBClassifier model
    - Predict labels of all images in a folder (using the param and image model and compare)
    - Predict new labels that will give good images

## Usage

Installing all the requirements:
```
$ cat requirements.txt
torch
torchvision
pillow
numpy
hpc-campaign
adios2
python-dateutil
matplotlib
joblib
scikit-learn
xgboost

$ ~/_penv/ai-adios-penv/bin/pip install -r requirements.txt
$ du -h -d 1 ~/_penv/ai-adios-penv/
912M	~/_penv/ai-adios-penv/
```

OpenMP needs to be installed for the XGBClassifier, and I needed to do:
```
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
```

### Step 1: Visual transformer for labeling images

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

### Step 1: XGBClassifier for labeling images

Usage:
```
$ python pre-train.py -h
usage: pre-train.py [-h] [--out-model OUT_MODEL] folder

positional arguments:
  folder                Folder with all pre-labeled images

options:
  -h, --help            show this help message and exit
  --out-model OUT_MODEL
                        Output pre-trained model

$ python pre-train.py smallspace/img/
Train an XGBClassifier on 236 samples and 79 testing samples
Accuracy: 0.9493670886075949
              precision    recall  f1-score   support

           0       0.95      0.98      0.96        55
           1       0.95      0.88      0.91        24

    accuracy                           0.95        79
   macro avg       0.95      0.93      0.94        79
weighted avg       0.95      0.95      0.95        79

Saved model to param_initial.json
```

Sanity check, predict all the images in the small dataset:
```
$ python predict_exising_images.py smallspace.aca image_model/image_model.pth param_initial.json
```

### Step 2: Create a campaign with Gray-Scott runs

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
     1. /Users/user/_projects/adios/ISC-tutorial/bigspace

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

### Step 2: Create a campaign with images

Create an archive file in the . folder.

```
python create_small_archive.py smallspace/img smallspace.aca
```

### Step 3: Train XGBClassifier using big ADIOS runs

The code will loop a max number of iterations, choose random batched of runs from the big campaign of size given by the input parameter, read the ADIOS data, generate images, use the image model to label the images and train on the batch. If the accuracy is bettern than the given parameter the process stops.

<img width="849" height="393" alt="Train loop for the XGBClassifier" src="https://github.com/user-attachments/assets/8998a4d8-68fe-4bae-a6a3-55427c7c5edc" />

If a pre-trained model is available (default this is set to None) the training starts with this model.

```
$ python train.py -h
usage: train.py [-h] [--img-model IMG_MODEL] [--param-model PARAM_MODEL] [--batch BATCH] [--iterations ITERATIONS] [--accuracy ACCURACY] [--verbose VERBOSE] campaign

positional arguments:
  campaign              Campaign file with all the Gray-Scott runs

options:
  -h, --help            show this help message and exit
  --img-model IMG_MODEL
  --param-model PARAM_MODEL
                        pre-trained initial parameted model
  --batch BATCH
  --iterations ITERATIONS
  --accuracy ACCURACY
  --verbose VERBOSE

$ python train.py bigspace/bigspace.aca --img-model image_model/image_model.pth --batch 200 --iterations 3 --param-model param_initial.json
Iteration 0
[0] 200 uncertain samples selected
!! Dataset Du0.353_Dv0.0404_F0.002438_k0.0502/gs/U has nan values, removing it from the batch
!! Dataset Du0.32_Dv0.0276_F0.0179_k0.0605/gs/U has nan values, removing it from the batch
!! Dataset Du0.379_Dv0.0238_F0.01327_k0.0476/gs/U has nan values, removing it from the batch
!! Dataset Du0.394_Dv0.0208_F0.0107_k0.0481/gs/U has nan values, removing it from the batch
!! Dataset Du0.362_Dv0.0195_F0.02673_k0.0579/gs/U has nan values, removing it from the batch
Training on 195 datasets. Generating images...
Train an XGBClassifier on 146 samples and 49 testing samples
Accuracy: 0.8979591836734694
              precision    recall  f1-score   support

           0       0.98      0.91      0.94        45
           1       0.43      0.75      0.55         4

    accuracy                           0.90        49
   macro avg       0.70      0.83      0.74        49
weighted avg       0.93      0.90      0.91        49

Iteration 1
[1] 200 uncertain samples selected
!! Dataset Du0.353_Dv0.0404_F0.002438_k0.0502/gs/U has nan values, removing it from the batch
!! Dataset Du0.394_Dv0.0208_F0.0107_k0.0481/gs/U has nan values, removing it from the batch
!! Dataset Du0.379_Dv0.0238_F0.01327_k0.0476/gs/U has nan values, removing it from the batch
!! Dataset Du0.32_Dv0.0276_F0.0179_k0.0605/gs/U has nan values, removing it from the batch
!! Dataset Du0.362_Dv0.0195_F0.02673_k0.0579/gs/U has nan values, removing it from the batch
!! Dataset Du0.132_Dv0.398_F0.0137_k0.0404/gs/U has nan values, removing it from the batch
Training on 194 datasets. Generating images...
Train an XGBClassifier on 291 samples and 98 testing samples
Accuracy: 0.9795918367346939
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        91
           1       0.78      1.00      0.88         7

    accuracy                           0.98        98
   macro avg       0.89      0.99      0.93        98
weighted avg       0.98      0.98      0.98        98

Saved model to param_model/param_xgb.json
```

### Step 4: Predict labels of existing runs using XGBClassifier and the Image model

Predict all labels of all images found in a given campaign. The code assumes the campaign stores the png images directly.

```
$ python predict_exising_images.py 
usage: predict_exising_images.py [-h] [--path PATH] campaign img_model params_model

positional arguments:
  campaign      Camaign file with images that will be predicted with both models
  img_model
  params_model

options:
  -h, --help    show this help message and exit
  --path PATH

$ python predict_exising_images.py smallspace.aca image_model/image_model.pth param_model/param_xgb.json
Dataset: (0.15, 0.15, 0.035, 0.05) ParamModel: bad ImgModel: bad
Dataset: (0.15, 0.15, 0.03, 0.06) ParamModel: bad ImgModel: bad
Dataset: (0.25, 0.05, 0.025, 0.045) ParamModel: bad ImgModel: bad
Dataset: (0.15, 0.15, 0.015, 0.065) ParamModel: bad ImgModel: bad
Dataset: (0.15, 0.1, 0.03, 0.065) ParamModel: bad ImgModel: bad
Dataset: (0.2, 0.05, 0.015, 0.06) ParamModel: good ImgModel: good
Dataset: (0.25, 0.05, 0.035, 0.065) ParamModel: good ImgModel: good
Dataset: (0.25, 0.1, 0.015, 0.06) ParamModel: bad ImgModel: good
Dataset: (0.2, 0.05, 0.015, 0.055) ParamModel: good ImgModel: good
Dataset: (0.25, 0.1, 0.005, 0.055) ParamModel: bad ImgModel: good
Dataset: (0.15, 0.05, 0.02, 0.055) ParamModel: good ImgModel: good
Dataset: (0.25, 0.05, 0.005, 0.06) ParamModel: good ImgModel: good
```

## Step 4: Find run parameters that will create good images

Given a range for each parameter, the code samples the space with a given frequency and returns all parameter combinations that could produce good images (with the associated confidence).

```
$ python predict_good_labels.py -h
usage: predict_good_labels.py [-h] [--Du-min DU_MIN] [--Du-max DU_MAX] [--Dv-min DV_MIN] [--Dv-max DV_MAX] [--F-min F_MIN] [--F-max F_MAX] [--k-min K_MIN] [--k-max K_MAX]
                              [--sampling-frequency SAMPLING_FREQUENCY] [--model MODEL]

options:
  -h, --help            show this help message and exit
  --Du-min DU_MIN       Gray-Scott Du parameter value
  --Du-max DU_MAX       Gray-Scott Du parameter value
  --Dv-min DV_MIN       Gray-Scott Dv parameter value
  --Dv-max DV_MAX       Gray-Scott Dv parameter value
  --F-min F_MIN         Gray-Scott F parameter value
  --F-max F_MAX         Gray-Scott F parameter value
  --k-min K_MIN         Gray-Scott k parameter value
  --k-max K_MAX         Gray-Scott k parameter value
  --sampling-frequency SAMPLING_FREQUENCY
                        Number of samples per parameter dimension. Example: sampling_frequency=10 evaluates 10^4 parameter combinations.
  --model MODEL
$ python predict_good_labels.py --sampling-frequency 4
{'Du': np.float32(0.16667), 'Dv': np.float32(0.06), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.9174705147743225}
{'Du': np.float32(0.16667), 'Dv': np.float32(0.04), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.8929235935211182}
{'Du': np.float32(0.16667), 'Dv': np.float32(0.02), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.8929235935211182}
{'Du': np.float32(0.23333), 'Dv': np.float32(0.06), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.8885332942008972}
{'Du': np.float32(0.16667), 'Dv': np.float32(0.06), 'F': np.float32(0.01833), 'k': np.float32(0.05), 'probability': 0.8866250514984131}
{'Du': np.float32(0.1), 'Dv': np.float32(0.06), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.8865153193473816}
{'Du': np.float32(0.1), 'Dv': np.float32(0.04), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.882911205291748}
{'Du': np.float32(0.1), 'Dv': np.float32(0.02), 'F': np.float32(0.025), 'k': np.float32(0.05), 'probability': 0.882911205291748}
{'Du': np.float32(0.16667), 'Dv': np.float32(0.06), 'F': np.float32(0.005), 'k': np.float32(0.05), 'probability': 0.8782230019569397}
{'Du': np.float32(0.3), 'Dv': np.float32(0.06), 'F': np.float32(0.025), 'k': np.float32(0.07), 'probability': 0.8630889058113098}
{'Du': np.float32(0.3), 'Dv': np.float32(0.06), 'F': np.float32(0.005), 'k': np.float32(0.05), 'probability': 0.8504162430763245}
{'Du': np.float32(0.3), 'Dv': np.float32(0.06), 'F': np.float32(0.025), 'k': np.float32(0.09), 'probability': 0.8487341403961182}
{'Du': np.float32(0.1), 'Dv': np.float32(0.06), 'F': np.float32(0.01833), 'k': np.float32(0.05), 'probability': 0.8484734892845154}
```
# ADC_Aware_Learning

Codes for “RedPIM: An Efficient PIM Accelerator Design with Reduced Analog-to-Digital Conversions”, Jiale Li, Yulin Fu, Sean Longyu Ma, Chiu-Wing Sham, Chong Fu 


## Structure
```
+---MNIST
|       models.py
|       search.py
|       test_acc.py
|       train.py
+---cifar10
|       models.py
|       search.py
|       searchResnet.py
|       test_acc.py
|       test_accResnet.py
|       train.py
|       trainResnet.py
|
+---cifar100
|       models.py
|       searchResnet.py
|       test_acc.py
|       test_accResnet.py
|       trainResnet.py
|
+---ImageNet
|       models.py
|       searchResnet34.py
|       test_accResnet34.py
|       trainResnet.py
|
|
+---quant
|       quant_module.py
|
\---utils
        torch_utils.py
        view_pt.py
        __init__.py
```

## Dependency
```
numpy==2.3.2
pretrainedmodels==0.7.4
torch==2.5.1
torchvision==0.20.1
tqdm==4.67.1
```

## Run

Please refer to the paper for details such as hyperparameters.

```
cd MNIST
mkdir results
mkdir weights
python search.py
cat results.csv
vim train.py  to modify --bita. The results from results.csv should be increased by 1.
python train.py```

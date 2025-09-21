# ADC_Aware_Learning

Codes for “[RedPIM: An Efficient PIM Accelerator Design with Reduced Analog-to-Digital Conversions](https://doi.org/10.1145/3769122) (pubilished at TODAES)”, Jiale Li, Yulin Fu, Sean Longyu Ma, Chiu-Wing Sham, Chong Fu 

RedPIM, an efficient ReRAM-based PIM accelerator design for deep neural networks (DNNs) that reduces the number of analog-to-digital conversions. RedPIM exploits the fact that in ReRAM-based PIM accelerators, the overall energy consumption generally increases with the number of activated analog-to-digital conversions. Specifically, we introduce a novel training algorithm that is aware of the ADC overhead during activation value quantization and optimizes accuracy concurrently. From a hardware design perspective, we develop a lookup table (LUT)-based quantization module to enable efficient and low-cost activation value quantization. In addition, we propose an efficient adaptive operation unit (OU) size assignment scheme that further minimizes analog-to-digital conversions by considering activation sparsity and weight distribution.

<img width="5437" height="941" alt="image" src="https://github.com/user-attachments/assets/1562b548-c109-4e45-8074-bc8b10c093e2" />

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

Please refer to the paper for details such as hyperparameters. Taking the running of the LeNet-5 neural network as an example.

1.Create the necessary directories
```
cd MNIST
mkdir results
mkdir weights
```

2.Run the search script
```
python search.py
```

3.View the search result
```
cat results.csv
```

4.Modify the hyperparameter in the training script. The results from results.csv should be increased by 1.
```
vim train.py
```

5.Run the training script
```
python train.py
```

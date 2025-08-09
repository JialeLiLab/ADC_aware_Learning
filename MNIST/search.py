import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import models
from test_acc import test

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomCrop(28, padding=4),  
    transforms.ToTensor(),                 
])


trainset = torchvision.datasets.MNIST(
    root='./dataset', 
    train=True,                                   
    download=True,                                
    transform=transform_train                    
)


def train():

    model = models.LeNet_MixQ(10) 
    model.to(device)

    results_file = 'results/%s.txt' % opt.name
    
    criterion = nn.CrossEntropyLoss()
    
    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    arch_optimizer = torch.optim.SGD(alpha_params, opt.lra, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs, eta_min=opt.lr*0.01) 
    arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        arch_optimizer, T_max=opt.epochs, eta_min=opt.lr*0.3)

    model.train()
    start_epoch, epochs = 0, opt.epochs
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=16)
    test_best_acc = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = macc = 0.
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            arch_optimizer.zero_grad()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            
            if opt.complexity_decay != 0 or opt.complexity_decay_trivial!=0:
                loss_complexity = opt.complexity_decay * model.complexity_loss() 
                loss += loss_complexity

            loss.backward()
            optimizer.step()
            arch_optimizer.step()
            
            mloss = (mloss*i + loss.item()) / (i+1)
            macc = (macc*i + correct/opt.batch_size) / (i+1)
            s = '%10s%10.2f%10.3g'%('%d/%d'%(epoch,epochs-1), macc*100, mloss)
            pbar.set_description(s)

        print('========= architecture =========')
        best_arch  = model.fetch_best_arch()
        besta_str = "".join([str(x) for x in best_arch["best_activ"]])
        print(f'best_activ: {best_arch["best_activ"]}')


        scheduler.step()
        arch_scheduler.step()

        results = test(model, device)
        with open(results_file, 'a') as f:
            f.write(s + '%10.2f%10.3g'% results + '\n')
        test_acc = results[0]
        test_best_acc = max(test_best_acc, test_acc)

        final_epoch = epoch == epochs-1
        if True or final_epoch:
            with open(results_file, 'r') as f:
                chkpt = {'epoch': epoch,
                            'training_results': f.read(),
                            'model': model.module.state_dict() if type(
                                model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'arch_optimizer': None if final_epoch else arch_optimizer.state_dict(),
                            'extra': {'time': time.ctime(), 'name': opt.name, 'besta': besta_str}}
            # Save last checkpoint
            torch.save(chkpt, wdir + '%s_last.pt'%opt.name)
            
            if test_acc == test_best_acc:
                torch.save(chkpt, wdir + '%s_best.pt'%opt.name)
    
    print('Finished Training')

    with open('results.csv', 'a') as f:
        print("mixed,%s,%d/%d, , , , ,%.1f,%.1f, ,%s"%
              (opt.name,epochs-1,epochs,macc*100,test_acc,
               besta_str), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25) 
    parser.add_argument('--batch-size', type=int, default=1024) 
    parser.add_argument('--device', default='6', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--name', default='mix_lenet_mnist_cd3e5', help='result and weight file name')
    parser.add_argument('--noshare', action='store_true', help='no share weight')
    parser.add_argument('--complexity-decay', '--cd', default=3e-5, type=float, metavar='W', help='complexity decay (default: 0)')
    parser.add_argument('--lra', '--learning-rate-alpha', default=0.1, type=float, metavar='LR', help='initial alpha learning rate')

    opt = parser.parse_args()
    print(opt)
    wdir = 'weights' + os.sep  # weights dir
    last = wdir + '%s_last.pt'%opt.name

    device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')


    train()

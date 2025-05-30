import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from data_loader.ImageNet_datasets import ImageNetData
import model.resnet_cbam as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange

def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def main(args):
    if 0 == len(args.resume):
        logger = Logger('./logs/'+args.model+'.log')
    else:
        logger = Logger('./logs/'+args.model+'.log', True)

    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    gpus = args.gpu.split(',')
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Loading datasets...")
    train_datasets = ImageNetData(
        img_root='./datasets',
        img_file='./datasets/train.txt',
        is_training=True,
        transform=data_transforms['train'],
        download=True
    )
    val_datasets = ImageNetData(
        img_root='./datasets',
        img_file='./datasets/val.txt',
        is_training=False,
        transform=data_transforms['val'],
        download=True
    )
    
    print(f"Train dataset size: {len(train_datasets)}")
    print(f"Validation dataset size: {len(val_datasets)}")
    
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size*len(gpus), shuffle=True, num_workers=8)
    val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size=1024, shuffle=False, num_workers=8)

    if args.debug:
        x, y = next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    print(f"Initializing model: {args.model}")
    if 'resnet50' == args.model.split('_')[0]:
        my_model = models.resnet50(pretrained=False)
    elif 'resnet50-cbam' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet50_cbam(pretrained=False)
    elif 'resnet101' == args.model.split('_')[0]:
        my_model = models.resnet101(pretrained=False)
    else:
        raise ModuleNotFoundError("Model not found")

    if is_use_cuda and 1 == len(gpus):
        my_model = my_model.cuda()
    elif is_use_cuda and 1 < len(gpus):
        my_model = nn.DataParallel(my_model.cuda())

    loss_fn = [nn.CrossEntropyLoss()]
    optimizer = optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) 
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    metric = [ClassErrorMeter([1, 5], True)]
    start_epoch = 0
    num_epochs = 90

    def train_epoch(model, dataloader, optimizer, loss_fn, metric, is_use_cuda):
        model.train()
        total_loss = 0.0
        
        for m in metric:
            m.reset()
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for i, (inputs, labels) in enumerate(pbar):
            if is_use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_fn[0](outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            metric[0].add(outputs.data, labels.data)
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{metric[0].value(1):.2f}%", "top5": f"{metric[0].value(5):.2f}%"})
        
        return total_loss / len(dataloader), metric

    def validate_epoch(model, dataloader, loss_fn, metric, is_use_cuda):
        model.eval()
        total_loss = 0.0
        
        for m in metric:
            m.reset()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation", leave=False)
            for i, (inputs, labels) in enumerate(pbar):
                if is_use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = model(inputs)
                loss = loss_fn[0](outputs, labels)
                
                metric[0].add(outputs.data, labels.data)
                total_loss += loss.item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{metric[0].value(1):.2f}%", "top5": f"{metric[0].value(5):.2f}%"})
        
        return total_loss / len(dataloader), metric
    
    print(f"Starting training for {num_epochs} epochs")
    for epoch in trange(start_epoch, num_epochs, desc="Epochs"):
        train_loss, train_metric = train_epoch(my_model, train_dataloaders, optimizer, loss_fn, metric, is_use_cuda)
        val_loss, val_metric = validate_epoch(my_model, val_dataloaders, loss_fn, metric, is_use_cuda)
        
        lr_schedule.step()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Top-1: {train_metric[0].value(1):.2f}%, Top-5: {train_metric[0].value(5):.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Top-1: {val_metric[0].value(1):.2f}%, Top-5: {val_metric[0].value(5):.2f}%")
        
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/top1', train_metric[0].value(1), epoch)
            writer.add_scalar('train/top5', train_metric[0].value(5), epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/top1', val_metric[0].value(1), epoch)
            writer.add_scalar('val/top5', val_metric[0].value(5), epoch)
        
        state = {
            'epoch': epoch + 1,
            'state_dict': my_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, f'checkpoints/{args.model}_epoch_{epoch+1}.pth')
    
    logger.append('Optimize Done!')


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-d', '--data_root', default='./datasets',
                         type=str, help='data root')
    parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
                         type=str, help='train file')
    parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
                         type=str, help='validation file')
    parser.add_argument('-m', '--model', default='resnet101',
                         type=str, help='model type')
    parser.add_argument('--batch_size', default=12,
                         type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    main(args)
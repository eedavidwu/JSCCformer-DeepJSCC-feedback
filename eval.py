import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
GPU_ids = [0]
#GPU_ids = [0,1]

import torch 
from Models.transformer_JSCC import JSCCModel
import torchvision
import torch
import torch.nn as nn 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import numpy as np
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device is " + str(device))
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def compute_AvePSNR(model,dataloader,snr,fb_snr):
    psnr_all_list = []
    model.eval()
    MSE_compute = nn.MSELoss(reduction='none')
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
        inputs = inputs.cuda()
        outputs= model(inputs,snr,fb_snr)
    
        MSE_each_image = (torch.sum(MSE_compute(outputs, inputs).view(b,-1),dim=1))/(c*h*w)
        PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
        one_batch_PSNR=PSNR_each_image.data.cpu().numpy()
        psnr_all_list.extend(one_batch_PSNR)
    Ave_PSNR=np.mean(psnr_all_list)
    return Ave_PSNR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Train:
    parser.add_argument("--best_ckpt_path", default='./ckpts/', type=str,help='best model path')
    parser.add_argument("--all_epoch", default='2000', type=int,help='Train_epoch')
    # Model and Channel:
    parser.add_argument("--model", default='SETR', type=str,help='Model select: SETR/ADSETR/')
    parser.add_argument("--rate", default=48, type=int,help='rate:8,16,32')
    parser.add_argument("--snr", default=10, type=int,help='awgn/slow fading/')
    parser.add_argument("--fb_snr", default=10, type=int,help='awgn/slow fading/')
    parser.add_argument("--refine", default=0,type=int, help='refine or not')
    args=parser.parse_args()
    model_path='./checkpoints_32/SNR_no_fb_tcn_'+str(args.rate)+'_snr_10.pth'
    one_snr_eval_flag=1


    print('head: 8')
    iter_num=1
    tcn=args.rate//iter_num
    print('iter:', iter_num)
    print('64*',tcn*iter_num)
    print('refine_flag:',args.refine)

    #print("16*24 1/8->(8,8) tcn=24")
    
    model = JSCCModel(patch_size=(4, 4), 
                    in_channels=3, 
                    out_channels=3, 
                    hidden_size=256, 
                    num_hidden_layers=8, 
                    num_attention_heads=8, 
                    intermediate_size=1024,
                    tcn=tcn,iteration=iter_num)
    #channel_fb_snr=args.fb_snr
    channel_fb_snr='perfect'

    #print(model)
    
    if len(GPU_ids)>1:
        model = nn.DataParallel(model,device_ids = GPU_ids)
    model = model.cuda()
    print(model)

    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    
    testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)
    loss_func = nn.MSELoss()
  
    checkpoint=torch.load(model_path)
    epoch_last=checkpoint["epoch"]
    model.load_state_dict(checkpoint["net"])
    best_psnr=checkpoint["Best_PSNR"]
    Trained_SNR=checkpoint['SNR']
    print("Load model:",model_path)
    print("Model is trained in SNR: ",Trained_SNR," with PSNR:",best_psnr," at epoch ",epoch_last)

    if one_snr_eval_flag==1:
        validate_snr=args.snr
        val_ave_psnr=compute_AvePSNR(model,testloader,validate_snr,channel_fb_snr)
        print('Evaluate in SNR ',validate_snr,' with performance:',val_ave_psnr)
    else:
        PSNR_list=[]
        for i in [-2,1,4,7,10,13]:
            validate_snr=i
            val_ave_psnr=compute_AvePSNR(model,testloader,validate_snr,channel_fb_snr)
            PSNR_list.append(val_ave_psnr)
            print('Evaluate in SNR ',i,' with performance:',val_ave_psnr)
        ave_PSNR=np.mean(PSNR_list)
        print('Evaluate in [-2,1,4,7,10,13]')
        print(PSNR_list)


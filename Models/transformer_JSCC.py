import logging
import math
import os
import numpy as np 

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from Models.transformer_model import TransConfig,TransModel2d,ReciverModel2d,Siam_linear
from torch.autograd import Variable
import math 
class Channel(nn.Module):
    def __init__(self):
        super(Channel, self).__init__()

    def forward(self, inputs,input_snr):
        in_shape=inputs.shape
        batch_size=in_shape[0]
        z=inputs.contiguous().view(batch_size,-1)
        dim_z=z.shape[1]//2
        real=z[:,:dim_z]
        imag=z[:,dim_z:]
        z_in=torch.complex(real,imag)

        ###power constraints:
        z_in=z_in.view(batch_size,-1)
        sig_pwr=torch.square(torch.abs(z_in))
        ave_sig_pwr=sig_pwr.mean(dim=1).unsqueeze(dim=1)
        z_in_norm=z_in/(torch.sqrt(ave_sig_pwr))

        ##awgn:
        noise_stddev=np.sqrt((10**(-input_snr/10))/2)
        noise_stddev_board=torch.from_numpy(noise_stddev).repeat(1,z_in_norm.shape[1]).cuda()
        mean=torch.zeros_like(noise_stddev_board).cuda()
        #compute noise:
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda())
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda())
        noise_complex=torch.complex(noise_real,noise_img)
        #add noise:
        z_out=z_in_norm+noise_complex
        real_out=torch.real(z_out)
        img_out=torch.imag(z_out)
        out=torch.cat((real_out,img_out),dim=1)
        channel_out=out.view(in_shape).float()
        return channel_out

class Fading_Channel(nn.Module):
    def __init__(self, args):
        super(Fading_Channel, self).__init__()

    def compensation(self,inputs,h_broadcast):
        h_norm_broadcast=torch.square(torch.abs(h_broadcast))
        h_conj_broadcast=torch.conj(h_broadcast)
        out=(h_conj_broadcast*inputs)/h_norm_broadcast
        return out
        
    def forward(self, inputs,h,input_snr):
        in_shape=inputs.shape
        batch_size=in_shape[0]

        ####Power constraint:
        z_in=inputs.view(batch_size,-1)
        sig_pwr=torch.square(torch.abs(z_in))
        ave_sig_pwr=sig_pwr.mean(dim=1).unsqueeze(dim=1)
        z_in_norm=z_in/(torch.sqrt(ave_sig_pwr))
        inputs_in_norm=z_in_norm.view(in_shape)
        ####Multipath Channel
        complex_para_out=h*inputs_in_norm

        ##awgn:
        noise_stddev=(np.sqrt(10**(-input_snr/10))/np.sqrt(2)).reshape(-1,1,1)
        noise_stddev_board=torch.from_numpy(noise_stddev).repeat(1,in_shape[1],in_shape[2]).cuda()
        mean=torch.zeros_like(noise_stddev_board).cuda()
        
        #compute ave:
        noise_real=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda())
        noise_img=Variable(torch.normal(mean=mean,std=noise_stddev_board).cuda())
        noise_complex=torch.complex(noise_real,noise_img)
        
        ##y=hx+w
        channel_out=complex_para_out+noise_complex

        #compensation:
        channel_com_out=self.compensation(channel_out,h)

        #new_noise=self.compensation(noise_complex,h_broadcast)
        #new_out=inputs_in_norm+new_noise

        return channel_com_out#,inputs_in_norm

class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, tcn,iteration):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config,tcn,iteration)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        #sample_rate=4,sample_v=16
        self.final_dense = nn.Linear(config.hidden_size,  tcn)
        #self.final_dense=Siam_linear(config,config.hidden_size, tcn)
        ##linear:x hidden-> 8*8*hidden/16/16
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v
        self.tcn=tcn

    def forward(self, x,feedback):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]

       
        hh = h // p1 
        ww = w // p2 

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2,h = hh, w = ww)
        x_in=torch.cat((x,feedback),dim=2)
        
        encode_x = self.bert_model(x_in)[-1]
        x_sequence = self.final_dense(encode_x)
        #x = rearrange(x_f, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c =tcn)
        #x_map = rearrange(x_sequence, "b (h w) (c) -> b c (h) (w)", h = hh, w = ww, c =self.tcn)
        return x_sequence 

class Decoder2D_trans(nn.Module):
    def __init__(self, config: TransConfig, tcn,iteration):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = ReciverModel2d(config,tcn*iteration)
        #sample_rate=4,sample_v=16
        #self.final_dense = nn.Linear(config.hidden_size, 192)
        self.final_dense = nn.Linear(config.hidden_size, 48)
        #self.final_dense =Siam_linear(config,config.hidden_size,48)

        ##linear:x hidden-> 8*8*hidden/16/16
        self.patch_size = config.patch_size


    def forward(self, x):
        ## x:(b, path_num, c)
        encode_x = self.bert_model(x)[-1] 
        #x = torch.sigmoid(self.final_dense(encode_x))   
        x = self.final_dense(encode_x)      
        #x_out = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = 8, p2 = 8, h = 4, w = 4, c =3)
        #x_out = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = 4, p2 = 4, h = 8, w = 8, c =3)

        return x 

class FL_De_Module(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding,out_padding,activation=None):
        super(FL_De_Module, self).__init__()
        self.Deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,output_padding=out_padding)
        self.GDN = nn.BatchNorm2d(out_channels)
        if activation=='sigmoid':
            self.activate_func=nn.Sigmoid()
        elif activation=='prelu':
            self.activate_func=nn.PReLU()
        elif activation==None:
            self.activate_func=None            

    def forward(self, inputs):
        out_deconv1=self.Deconv1(inputs)
        out_bn=self.GDN(out_deconv1)
        if self.activate_func != None:
            out=self.activate_func(out_bn)
        else:
            out=out_bn
        return out


class JSCCModel(nn.Module):
    def __init__(self, patch_size=(32, 32), 
                        in_channels=3, 
                        out_channels=1, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        max_position_embeddings=64,
                        intermediate_size=512,
                        sample_rate=2,tcn=4,iteration=4):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            intermediate_size=intermediate_size,
                            max_position_embeddings=max_position_embeddings,
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config,tcn,iteration)
        #self.decoder_2d = Decoder2D(in_channels=tcn, out_channels=config.out_channels, features=decode_features)
        #self.res_decoder=Decoder_Res(in_channel=tcn*3)
        self.decoder_tran=Decoder2D_trans(config,tcn,iteration)
        self.channel=Channel()
        self.last_iter=iteration
        #feature each iteration
        self.tcn=tcn
        self.fb_num=tcn
        self.iteration=iteration

    def transmit_feature(self,feature,channel_snr):
        #feature_ave=torch.zeros_like(feature).float().cuda()
        channel_out=self.channel(feature,channel_snr)
        return channel_out

    def forward(self, x,input_snr,fb_snr):
        batch_size=x.shape[0]
        if input_snr=='random':
            snr=np.random.rand(batch_size,)*(10+2)-2
        else:
            snr=np.broadcast_to(input_snr,(batch_size,1))
        tcn=self.tcn
        feedback_for_encoder_all=torch.zeros(batch_size,64,(self.fb_num)*(self.last_iter-1)).cuda()
        latent_for_decoder_all=torch.zeros(batch_size,64,self.tcn*(self.last_iter)).cuda()
        #latent_for_decoder_fb=torch.zeros(batch_size,64,self.tcn*(self.last_iter)).cuda()
        
        final_z_seq = self.encoder_2d(x,feedback_for_encoder_all)
        #channel_out=self.transmit_feature(final_z_seq,snr)
        channel_out=self.channel(final_z_seq,snr)
        #channel_out=final_z_seq
        latent_for_decoder_all[:,:,0*tcn:(0+1)*tcn]=channel_out
        out=self.decoder_tran(latent_for_decoder_all)
        x_out = rearrange(out, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = 4, p2 = 4, h = 8, w = 8, c =3)
        return x_out


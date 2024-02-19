
'''
replace batch norm with group norm

'''
import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.autograd.variable as Variable
import torch.distributed as dist

def weight_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1, activation="swish"):
        super(conv_block, self).__init__()
        if activation == "relu":
            self.acti = nn.ReLU(True)
        elif activation == "lrelu":
            self.acti = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "swish":
            self.acti = nn.SiLU(True)
        
        self.group_norm = GroupNorm(ch_out)
        self.conv = nn.Sequential( 
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),
            self.group_norm,
            self.acti)
    def forward(self, x):
        out = self.conv(x)
        return out


class pre_vq_conv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=0):
        super(pre_vq_conv, self).__init__()
        
        #self.group_norm = GroupNorm(ch_out)
        self.conv = nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p)
    def forward(self, x):
        out = self.conv(x)
        return out


class post_vq_conv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=0):
        super(post_vq_conv, self).__init__()
        
        #self.group_norm = GroupNorm(ch_out)
        self.conv = nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p)
    def forward(self, x):
        out = self.conv(x)
        return out


class ResNet_block(nn.Module):
    "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"
    def __init__(self, ch, k_size, stride=1, p=1, activation="relu", include_last_bnact = True):
        super(ResNet_block, self).__init__()
        if activation == "relu":
            self.acti = nn.ReLU(True)
        elif activation == "lrelu":
            self.acti = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "swish":
            self.acti = nn.SiLU(True) #Swish()

        self.group_norm = GroupNorm(ch)
        self.conv = []
        self.conv.append(nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p))
        self.conv.append(self.group_norm) #nn.BatchNorm3d(ch))
        self.conv.append(self.acti)
        self.conv.append(nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p))
        #self.conv.append(nn.BatchNorm3d(ch))
        if include_last_bnact:
            self.conv.append(self.group_norm) #nn.BatchNorm3d(ch))
            self.conv.append(self.acti)
        
        self.conv_module = nn.Sequential(*self.conv)
        
    def forward(self, x):
        out = self.conv_module(x) + x
        return out


class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
    def __init__(self, ch_in, ch_out, k_size=1, scale=2, padding = 1, align_corners=False):
        super(up_conv, self).__init__()
        self.group_norm = GroupNorm(ch_out)
        self.acti = nn.ReLU(True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale),
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride = 1, padding = padding),
            #nn.BatchNorm3d(ch_out),
            self.group_norm,
            #nn.ReLU(inplace = True),
            self.acti)
    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    """ Encoder module """
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv_block(ch_in=1, ch_out=32, k_size=4, stride=2, activation="lrelu")
        self.res_block1 = ResNet_block(ch=32, k_size=3, activation="lrelu")
        #self.MaxPool1 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv2 = conv_block(ch_in=32, ch_out=64, k_size=4, stride=2, activation="lrelu")
        self.res_block2 = ResNet_block(ch=64, k_size=3, activation="lrelu")
        #self.MaxPool2 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv3 = conv_block(ch_in=64, ch_out=128, k_size=4, stride=2, activation="lrelu")
        self.res_block3 = ResNet_block(ch=128, k_size=3, activation="lrelu")
        #self.MaxPool3 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv4 = conv_block(ch_in=128, ch_out=256, k_size=4, stride=2, activation="lrelu")
        self.res_block4 = ResNet_block(ch=256, k_size=3, activation="lrelu")

        self.conv5 = conv_block(ch_in=256, ch_out=512, k_size=3, stride=1, activation="lrelu")
        self.res_block5 = ResNet_block(ch=512, k_size=3, activation="lrelu")
        #self.MaxPool4 = nn.MaxPool3d(3, stride=2, padding=1)

        #self.reset_parameters()
      
    # def reset_parameters(self):
    #     for weight in self.parameters():
    #         stdv = 1.0 / math.sqrt(weight.size(0))
    #         torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res_block1(x1)
        #print("x1 shape", x1.shape)
        #x1 = self.MaxPool1(x1) 
        
        x2 = self.conv2(x1)
        x2 = self.res_block2(x2)
        #x2 = self.MaxPool2(x2) 
        #print("x2 shape", x2.shape)

        x3 = self.conv3(x2)
        x3 = self.res_block3(x3)
        #x3 = self.MaxPool3(x3) 
        #print("x3 shape", x3.shape)
        
        x4 = self.conv4(x3)
        x4 = self.res_block4(x4) 
        #print("x4 shape", x4.shape)
        
        x5 = self.conv5(x4)
        x5 = self.res_block5(x5) # [batch_size, 512, 8, 8, 8]
        return x5#, latent_mean, latent_var


class Decoder(nn.Module):
    """ Decoder Module """
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        # self.latent_dim = latent_dim
        #self.linear_up = nn.Linear(latent_dim, 256*6*11*11)
        # self.relu = nn.ReLU()
        
        self.c, self.h, self.w = z_dim
        self.upsize4 = up_conv(ch_in=512, ch_out=256, k_size=(3,3,3), scale=2) # 8
        self.res_block4 = ResNet_block(ch=256, k_size=3, activation="relu")
        self.upsize3 = up_conv(ch_in=256, ch_out=128, k_size=(3,3,3), scale=2) # 16
        self.res_block3 = ResNet_block(ch=128, k_size=3, activation="relu")        
        self.upsize2 = up_conv(ch_in=128, ch_out=64, k_size=3, scale=2) # 32
        self.res_block2 = ResNet_block(ch=64, k_size=3, activation="relu")   
        self.upsize1 = up_conv(ch_in=64, ch_out=32, k_size=(3,3,3), scale=2) # 64
        self.res_block1 = ResNet_block(ch=32, k_size=3, activation="relu")

        self.upsize0 = nn.Upsample(scale_factor=2) # 128
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)

        self.final_acti = nn.Tanh()

    #     self.reset_parameters()
      
    # def reset_parameters(self):
    #     for weight in self.parameters():
    #         stdv = 1.0 / math.sqrt(weight.size(0))
    #         torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        #x = self.linear_up(x)
        # x4_ = self.relu(x4_)

        
        x = x.view(-1, 512, self.c, self.h, self.w)
        x = self.upsize4(x) # 256 16x16x16
        #print("up conv4", x4_.shape)
        x = self.res_block4(x)
        #print("res conv4", x4_.shape)

        x = self.upsize3(x) # 128 32x32x32
        #print("up conv3", x3_.shape)        
        x = self.res_block3(x)
        #print("res conv3", x3_.shape)

        x = self.upsize2(x) # 64 64x64x64
        #print("up conv2", x2_.shape)           
        x = self.res_block2(x)
        #print("res conv2", x2_.shape)

        x = self.upsize1(x) # 128 128*128*128
        #print("up conv1", x1_.shape)   
        x = self.res_block1(x)
        
        #x = self.upsize0(x)
        x = self.final_conv(x)
        #print("dec res conv1", x1_.shape)   

        x = self.final_acti(x)

        return x

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = 1 #int(np.ceil((kw-1.0)/2))
        #print(padw)
        # 1 -> 64
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, 
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf # nf = 64
        norm_layer = GroupNorm
        print("dis norm layer reinitialized to group norm")
        for n in range(1, n_layers+1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=4,
                                stride=1, padding=0)]]
        
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=True, restart_thres=1.0):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        #self.embed = nn.Embedding(n_codes, embedding_dim)

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = z.view(-1, z.shape[1])#shift_dim(z, 1, -1).flatten(end_dim=-2)
        print("flat_inputs:", flat_inputs.shape)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = z.view(-1, z.shape[1]) #shift_dim(z, 1, -1).flatten(end_dim=-2)  # [bthw, c]
        #print("self.embeddings: ", self.embeddings.shape)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
            - 2 * flat_inputs @ self.embeddings.t() \
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)  # [bthw, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(
            flat_inputs)  # [bthw, ncode]
        encoding_indices = encoding_indices.view(
            z.shape[0], *z.shape[2:])  # [b, t, h, w, ncode]
        #print("encoding indices: ", encoding_indices.shape)
        embeddings = F.embedding(
            encoding_indices, self.embeddings)  # [b, t, h, w, c]
        
        #embeddings = self.embed(encoding_indices)
        embeddings = embeddings.permute((0, 4, 1, 2, 3)).contiguous() #shift_dim(embeddings, -1, 1)  # [b, c, t, h, w]
        #print("embeddings: ", embeddings.shape)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1)
                         >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
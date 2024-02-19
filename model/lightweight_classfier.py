# a light weight classifier to compute the important region that can be used in MLM in stage 2

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups=32):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(in_channels)
        self.q = nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
    def forward(self, x, **ignore_kwargs):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,d,h,w = q.shape
        q = q.reshape(b,c,d*h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,d*h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,d*h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,d,h,w)

        h_ = self.proj_out(h_)

        return x+h_, w_
    
class socre_net(nn.Module):
    def __init__(self,
                 input_dim,
                 n_classes,
                 interm_linear_dim = 2048,
                 score_pred_net_mode = "2layer_vanilla",
                 patch_size = 16,
                 ):
        super().__init__()
        self.hwd = 8 #int(input_token_num**(1/3)) # cubic root of h*w*d
        self.att = AttnBlock(input_dim)
        self.groupnorm1 = GroupNorm(channels=1, num_groups=1)
        self.linear_in = input_dim * self.hwd * self.hwd * self.hwd
        self.patch_size = patch_size
        assert self.patch_size * self.hwd == 128, "the current lat dim is {}, the default image size is 128, patch size is {} which is not divisible".format(self.hwd, self.patch_size)
        self.linear_interm = interm_linear_dim
        self.score_pred_net_mode = score_pred_net_mode
        self.group_norm_att = GroupNorm(input_dim)
        # compute score
        if score_pred_net_mode == "2layer_vanilla":
            self.score_pred_net = nn.Sequential(
                nn.LayerNorm([input_dim, self.hwd, self.hwd, self.hwd], elementwise_affine=False),
                #nn.BatchNorm3d(input_dim),
                nn.Conv3d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(input_dim, 1, kernel_size=1, stride=1, padding=0),
                #nn.BatchNorm3d(1),
                self.groupnorm1,
                nn.Sigmoid(),
            )
        elif score_pred_net_mode == "2layer_att":
            self.score_pred_net = nn.Sequential(
                self.group_norm_att,
                nn.ReLU(),
            )    
            
        else:    
            raise NotImplementedError("score_pred_net_mode: {} not implemented".format(score_pred_net_mode))

        self.n_classes = n_classes
        self.group_norm = GroupNorm(self.linear_interm)
        print("linear in: ", self.linear_in)
        self.classifier = nn.Sequential(
            #self.group_norm,
            nn.Linear(self.linear_in, self.linear_interm),
            self.group_norm,
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.linear_interm, self.n_classes)
        )
        self.norm_feature = nn.LayerNorm([input_dim, self.hwd, self.hwd, self.hwd], elementwise_affine=False)
        #self.norm_feature = nn.BatchNorm3d(input_dim)

    
    def forward(self, image_features, label = None):
        # b, c, d, h, w 3d med images
        batch_size, channel, depth, height, width = image_features.size()
        # now channel is 1 after score net
        if self.score_pred_net_mode == "2layer_vanilla":
            pred_score = self.score_pred_net(image_features) # get importance score
            img_feat_norm = self.norm_feature(image_features) # get normalized feature
        
            img_feat_weighted = img_feat_norm * pred_score # get weighted feature
            #print("img_feat_weighted shape: ", img_feat_weighted.shape)
            img_feat_class = img_feat_weighted.view(batch_size, -1)
            #print("img_feat_class shape: ", img_feat_class.shape)
            class_out = self.classifier(img_feat_class) # learn weighted feature by doing classification
            class_prob = F.softmax(class_out, dim=1)
            #print("class_prob shape: ", class_prob.shape)
            pred_vis = F.interpolate(pred_score, scale_factor=self.patch_size, mode="nearest")
            
            if label is not None:
                label = label.long().squeeze(-1)
                #print("label shape: ", label.shape)
                loss = F.cross_entropy(class_out, label, label_smoothing=0.1)
                return loss, class_prob, pred_score, pred_vis, img_feat_weighted
            
        elif self.score_pred_net_mode == "2layer_att":
            print("============================================")
            print("=====THIS MODULE: {} IS NOT USED AND HAS NOT BEEN TESTED YET=====".format(self.score_pred_net_mode))
            print("============================================")
            img_feat_att, att = self.att(img_feat_norm)
            img_feat_att = self.score_pred_net(img_feat_att)
            img_feat_att = img_feat_att.view(batch_size, -1)
            class_out = self.classifier(img_feat_att)
            class_prob = F.softmax(class_out, dim=1)
            if label is not None:
                loss = F.cross_entropy(class_prob, label)
                return loss, class_prob, att, None, None
        
        return pred_score, pred_vis, img_feat_weighted
        
        
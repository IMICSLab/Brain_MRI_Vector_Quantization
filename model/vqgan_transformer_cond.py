# mask block, but keep the start index of each feature map in raster scan order

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import os

from model.transformer import GPT, sample_with_past
from model.autoencoder_128_cond_model import Encoder, Decoder, pre_vq_conv, post_vq_conv
from model.norm_ema_codebook import NormEMAVectorQuantizer
from model.lightweight_classfier import *


class LabelEnc(nn.Module):
    """ label encoder for class-conditional transformer"""
    def __init__(self, n_classes, quantize_interface=True):
        super(LabelEnc, self).__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c):
        c = c[:,None]
        if self.quantize_interface:
            return c, c.long()
        return c


def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        #print(top_k)
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            #print(top_k)
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits


class VQGANTransformer(nn.Module):
    def __init__(self, class_num, z_dim, codebook_dim, n_codes, mask_ratio, topk_ratio, model_weights, device):
        super(VQGANTransformer, self).__init__()

        self.device = device
        self.sos_token = 0 # default
        self.cond_class_dim = class_num # number of classes
        self.n_codes = n_codes
        self.hwd = z_dim[0]
        self.codebook_dim = codebook_dim
        weights_dict = torch.load(model_weights)
        self.topk_ratio = topk_ratio
        self.vqgan_encoder = Encoder().to(device)
        self.vqgan_encoder.load_state_dict(weights_dict["Encoder"])
        self.vqgan_encoder.eval()
        
        self.vqgan_decoder = Decoder(z_dim).to(device)
        self.vqgan_decoder.load_state_dict(weights_dict["Decoder"])
        self.vqgan_decoder.eval()
        
        self.prev_conv = pre_vq_conv(512, self.codebook_dim, 1).to(device)
        self.prev_conv.load_state_dict(weights_dict["preV_conv"])
        self.prev_conv.eval()
        
        self.postV_conv = post_vq_conv(self.codebook_dim, 512, 1).to(device)
        self.postV_conv.load_state_dict(weights_dict["postV_conv"])
        self.postV_conv.eval()
        
        # default to 2 layer vanilla
        self.score_net = socre_net(input_dim = self.codebook_dim, n_classes = class_num, interm_linear_dim = 1024, score_pred_net_mode = "2layer_vanilla").to(device)
        self.score_net.load_state_dict(weights_dict["ScoreNet"])
        self.score_net.eval()
        
        self.codebook = NormEMAVectorQuantizer(self.n_codes, self.codebook_dim).to(device) #Codebook(64, 512).to(device)
        self.codebook.load_state_dict(weights_dict["Codebook"])
        self.codebook.eval()
        
        self.block_size = self.n_codes
        self.gpt_vocab_size = self.n_codes + self.cond_class_dim + 1 # 1 for mask token
        self.transformer_config = {
            "vocab_size": self.gpt_vocab_size,
            "block_size": self.block_size,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**self.transformer_config).to(self.device)
        
        # keep 50%, mask 50%
        self.pkeep = 1-mask_ratio
        self.class_encoder = LabelEnc(n_classes=self.cond_class_dim).to(self.device)
        
        self.linear_prob_head = nn.Linear(self.gpt_vocab_size, self.cond_class_dim).to(self.device) # not used
        self.mask_token = self.n_codes # [MASK] token
        self.codebook_and_mask_dim = self.n_codes + 1 # this should be 0~codebook codes-1 for codebook size, codebook codes for mask token, codebook codes + 2 for two classes

    @torch.no_grad()
    def encode_to_z(self, x): # x: image
        im_enc = self.vqgan_encoder(x)
        #print("im_enc shape: ", im_enc.shape)
        enc_to_vq = self.prev_conv(im_enc)
        pred_score, _, _ = self.score_net(enc_to_vq, label = None)
        vq_out = self.codebook(enc_to_vq)
        quant_z = vq_out["embeddings"]
        indices = vq_out["encodings"]
        #quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return im_enc, quant_z, indices, pred_score

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, indices = self.class_encoder.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices
    
    @torch.no_grad()
    def z_to_image(self, indices):
        #ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = F.embedding(indices, self.codebook.embedding.weight).reshape(indices.shape[0], self.hwd, self.hwd, self.hwd, self.codebook_dim)
        ix_to_vectors = self.postV_conv(ix_to_vectors.permute(0,4,1,2,3).contiguous())#.permute(0, 4, 1, 2, 3) #ix_to_vectors.permute(0, 3, 1, 2)
        assert ix_to_vectors.shape[1:] == (512,self.hwd,self.hwd,self.hwd), "something wrong with the shape"
        image = self.vqgan_decoder(ix_to_vectors)
        return ix_to_vectors, image

    def tensor_to_list(self, t):
        t = t.detach().cpu().numpy()
        return t.tolist()
        
    def greedy_mask(self, importance_mask, codes_per_chunk):
        '''
        greedly mask the token based on importance score, with its neighbor token in spatial and temporal dimension
        '''
        bs, codes = importance_mask.shape
        codes_per_feature_map = codes_per_chunk ** 2
        t = self.tensor_to_list(importance_mask)
        res = []
        for i in range(bs):
            temp = []
            tt = t[i]
            j = 0
            while tt:
                mi = tt[j]
                rand_int = torch.randint(0, 2, size=(1,)).item()
                #print(rand_int)
                temp.append(mi)
                if rand_int == 0:
                    # can easily adapt to mask left and right (both direction), rather than single direction to the right
                    if mi % codes_per_feature_map > 2 and mi % codes_per_feature_map < codes_per_feature_map - 2:
                        temp.append(mi+1)
                    else:
                        temp.append(mi-1)
                else:
                    if mi + codes_per_feature_map < codes_per_chunk * codes_per_feature_map:
                        temp.append(mi+codes_per_feature_map)
                    else:
                        temp.append(mi-codes_per_feature_map)
                if mi in tt:
                    tt.remove(mi)
                if mi+1 in tt:
                    tt.remove(mi+1)
                if mi+codes_per_feature_map in tt:
                    tt.remove(mi+codes_per_feature_map)
                if mi-1 in tt:
                    tt.remove(mi-1)
                if mi-codes_per_feature_map in tt:
                    tt.remove(mi-codes_per_feature_map)
            temp = list(set(temp)) # double check for duplicate
            res.append(temp)
        return res
        
    
    def find_diff_indices(self, importance_mask, total):
        res = []
        bs, codes = total.shape
        for i in range(bs):
            total_i = total[i].detach().cpu().numpy()
            importance_mask_i = importance_mask[i].detach().cpu().numpy()
            diff = np.setdiff1d(total_i, importance_mask_i).tolist()
            res.append(diff)
        return torch.Tensor(res).to(total.device)
        
        
    def mask_block(self, x, importance_score, mask_ratio, top_ratio, block_size = 2):
        '''
        top_ratio: how much are important codes
        '''
        # x: (bs, 512)
        # importance_score: (bs,1, 8,8,8)
        bs, codes = x.shape
        codes_per_chunk = 8
        codes_per_feature_map = codes_per_chunk ** 2
        #print("codes per chunk: ", codes_per_chunk)
        link_codes = [] # codes linked between each feature map should not be masked, perserve continuity between slices
        for i in range(0, codes, int(codes_per_chunk)**2):
            if i != 0:
                link_codes.append(i)
                link_codes.append(i-1)
        #print("link_codes: ", link_codes)
        # depth-agnostic mask block based on importance score
        importance_score_clone = importance_score.clone()
        importance_score_clone = importance_score_clone.view(bs, -1) # bs, 512 
        importance_score_clone[:,link_codes] = -float("Inf") # set the link codes to 0, don't mask them
        sort_score, sort_order = importance_score_clone.sort(descending=True,dim=1)
        # from here we are working on the actual indices
        total_indices_to_mask = int(codes * mask_ratio)
        importance_mask_num = total_indices_to_mask // 2 # hybrid, important half, unimportant half
        importance_start_indices_num = importance_mask_num // block_size # becasue for each important indices, we want to mask a block around it, default block size is 1*2*2
        top_important_indices = sort_order[:,:int(codes * top_ratio)] # set top_ratio relatively high to avoid duplicate
        #random_sample_tii = torch.randint(0, top_important_indices.shape[1], size=(bs, importance_start_indices_num)) # random sample indices from top important indices
        random_sample_tii = torch.multinomial(top_important_indices.float(), num_samples=importance_start_indices_num, replacement=False) # return index
        temp_importance_mask = top_important_indices.gather(1, random_sample_tii) # get the actual top important indices to be masked
        #print("temp importance_mask: ", temp_importance_mask, "shape: ", temp_importance_mask.shape)
        # get the block indices, 1*2*2, indices in the importance_mask is the upper left corner of the block
        importance_mask = temp_importance_mask.clone()
        #print("importance_mask shape: ", importance_mask.shape)
        importance_mask = self.greedy_mask(importance_mask, codes_per_chunk) # greedy mask, either mask in spatial or depth dimension (sparial-depth agnostic)
        for i in range(bs):
            left = importance_mask_num - len(importance_mask[i])
            bl = sort_order[i,int(codes * top_ratio):].detach().cpu().numpy().tolist()
            for j in range(left):
                for ind in bl:
                    if ind in importance_mask[i]:
                        continue
                    else:
                        importance_mask[i].append(ind)
                        bl.remove(ind)
                        break

        importance_mask = torch.from_numpy(np.array(importance_mask)).to(self.device)
        assert importance_mask.shape[1] == importance_mask_num, "importance_mask shape not correct, got {}".format(importance_mask.shape)
        
        # get the acutal index in the score_order
        # block_indices_im = torch.stack(block_indices_im).view(bs, -1)
        # importance_mask = sort_order.gather(1, block_indices_im)
        # print("importance_mask: ", importance_mask)
        # print("importance_mask shape: ", importance_mask.shape)
        
        # get set difference
        un_importance_mask_num = total_indices_to_mask - importance_mask_num
        unimportance_indices = self.find_diff_indices(importance_mask, sort_order)

        #unimportance_indices = sort_order[:,int(codes * 0.5):] # let the last half be unimportant
        random_sample_unim = torch.multinomial(unimportance_indices.float(), num_samples=un_importance_mask_num, replacement=False) #torch.randint(0, unimportance_indices.shape[1], size=(bs, un_importance_mask_num))
        unimportance_mask = unimportance_indices.gather(1, random_sample_unim)
        
        # double check for overlap between importance and unimportance mask
        overlap = torch.sum(torch.eq(importance_mask, unimportance_mask), dim=1)
        assert torch.any(overlap).item() == False, "overlap between importance and unimportance mask! overlap: {}".format(overlap) 
        #print("overlap: ", overlap)
        return importance_mask, unimportance_mask
        
        
    def forward(self, x, c):
        # quant z is actually im_enc
        quant_z, _, z_indices, importance_score = self.encode_to_z(x)
        #print(z_indices)
        #print("importance_score shape: ", importance_score.shape)
        _, c_indices = self.encode_to_c(c)
        c_indices += self.codebook_and_mask_dim # instead of we shift the whole codes in the codebook, we shift the class token, 1025,1026 rather than 0,1
        #print("c_indices shape: ", c_indices.shape)
        #z_indices = z_indices + self.cond_class_dim # 2: class condition dim, 2 classes

        if self.pkeep < 1.0:
            importance_mask, unimportance_mask = self.mask_block(z_indices, importance_score, mask_ratio=self.pkeep, top_ratio=self.topk_ratio, block_size=2)
            # replace importance_mask with [MASK] token
            temp_mask1 = torch.ones_like(z_indices)
            for i in range(z_indices.shape[0]):
                temp_mask1[i,importance_mask[i].long()] = 0
            temp_ind = temp_mask1 * z_indices + (1 - temp_mask1) * self.mask_token # hybrid is the key, important -> mask token, unimportant -> random token, prevent shortcut learning
            # replace unimportance_mask with random token
            temp_mask2 = torch.ones_like(z_indices)
            for i in range(z_indices.shape[0]):
                temp_mask2[i,unimportance_mask[i].long()] = 0
            random_indices = torch.randint_like(z_indices, low=0, high=self.n_codes)
            new_indices = temp_mask2 * temp_ind + (1 - temp_mask2) * random_indices
            
        else:
            new_indices = z_indices
        
        #print("new indices, ", new_indices)
        cz_indices = torch.cat((c_indices, new_indices), dim=1) # prepend class token

        target = z_indices
        #print("target, ", target)

        logits, _ = self.transformer(cz_indices[:, :-1])

        return quant_z, logits, target

    
    def linear_prob(self, logits):
        # ix_to_vectors = F.embedding(indices, self.codebook.embeddings).reshape(indices.shape[0], 8, 8, 8, 512)
        # ix_to_vectors = self.postV_conv(ix_to_vectors.permute(0,4,1,2,3).contiguous())#.permute(0, 4, 1, 2, 3) #ix_to_vectors.permute(0, 3, 1, 2)
        logits = torch.mean(logits[:, 1:, :], dim=1)
        class_logits = self.linear_prob_head(logits)
        return class_logits

        
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    def avoid_repeat_sampling(self, logits):
        batch_size = logits.size(0)
        out = logits.clone()
        for i in range(batch_size):
            for j in range(self.cond_class_dim):
                out[i, self.codebook_and_mask_dim + j] = -float('Inf')  # avoid sampling cls token
            out[i, self.mask_token] = -float('Inf')  # avoid sampling mask token
        return out
    
    
    @torch.no_grad()
    def sample(self, x, c, steps, top_k=None, top_p=None, temperature=1.0):
        self.transformer.eval()
        #print("sample c", c)
        x = torch.cat((c,x),dim=1)

        for k in range(steps):
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]  # crop context if needed
            logits, _ = self.transformer(x_cond)
            logits = logits[:, -1, :] / temperature
            logits = self.avoid_repeat_sampling(logits) # avoid sampling cls and mask token
            if top_k is not None:
                #logits = self.top_k_logits(logits, top_k)
                logits = top_k_top_p_filtering(logits, top_k, top_p)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:] # cut off condition
        #self.transformer.train()
        return x
        
        
    @torch.no_grad()
    def log_images(self, x, c):
        log = dict()
        with torch.no_grad():
            _, _, indices, importance_score = self.encode_to_z(x)
            _, c_indices = self.encode_to_c(c)
            c_indices += self.codebook_and_mask_dim

            start_indices = indices[:, :indices.shape[1] // 2]
            sample_indices = self.sample(start_indices, c_indices, steps=indices.shape[1] - start_indices.shape[1], top_k=None, top_p=None)
            # change back
            sample_indices = torch.clamp(sample_indices, min = 0, max = self.n_codes-1) # ideally there should not be any indices larger than n_codes or smaller than 0
            _,half_sample = self.z_to_image(sample_indices)

            start_indices = indices[:, :0]
            sample_indices = self.sample(start_indices, c_indices, steps=indices.shape[1], top_k=None, top_p=None)
            # change back
            sample_indices = torch.clamp(sample_indices, min = 0, max = self.n_codes-1)
            _,full_sample = self.z_to_image(sample_indices)

            _,x_rec = self.z_to_image(indices)
        
        log["class"] = c
        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.cat((x, x_rec, half_sample, full_sample), dim = 0)

    @torch.no_grad()
    def sample_inf(self, x, c, steps, top_k, top_p, temperature=1.0):
        x = torch.cat((c,x),dim=1)

        for k in range(steps):
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]  # crop context if needed
            logits, _ = self.transformer(x_cond)
            logits = logits[:, -1, :] / temperature
            logits = self.avoid_repeat_sampling(logits) # avoid sampling cls and mask token
            if top_k is not None:
                #logits = self.top_k_logits(logits, top_k)
                logits = top_k_top_p_filtering(logits, top_k, top_p)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:] # cut off condition
        #self.transformer.train()
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/szhou/pLGG")
import os

from model.transformer import GPT
from model.autoencoder_128 import *


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
    def __init__(self, z_dim, enc_dec_weights, device):
        super(VQGANTransformer, self).__init__()

        self.device = device
        self.sos_token = 0
        self.latent_dim = 8
        self.channels = 512 # codebook dim
        self.codes = self.latent_dim * self.latent_dim * self.latent_dim
        
        #weights_path = "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/BraTS_brainOnly/trail11_v4/pretrained_model"
        #weights_path = "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/BraTS_brainOnly/trail11_v4/pretrained_model"

        model_weights = enc_dec_weights #"/hpf/largeprojects/fkhalvati/Simon/3dgan_res/BraTS_brainOnly/paper_exp/trailvqgan_v3/pretrained_model/epoch_3999.pt" #os.path.join(weights_path, "epoch_3999.pt")

        self.vqgan_encoder = Encoder().to(device)
        self.vqgan_encoder.load_state_dict(torch.load(model_weights)["Encoder"])
        self.vqgan_encoder.eval()
        
        self.vqgan_decoder = Decoder(z_dim).to(device)
        self.vqgan_decoder.load_state_dict(torch.load(model_weights)["Decoder"])
        self.vqgan_decoder.eval()
        
        self.prev_conv = pre_vq_conv(self.channels, self.channels, 1).to(device)
        self.prev_conv.load_state_dict(torch.load(model_weights)["preV_conv"])
        self.prev_conv.eval()
        
        self.postV_conv = post_vq_conv(self.channels, self.channels, 1).to(device)
        self.postV_conv.load_state_dict(torch.load(model_weights)["postV_conv"])
        self.postV_conv.eval()
        
        self.codebook = Codebook(self.codes, self.channels).to(device)
        self.codebook.load_state_dict(torch.load(model_weights)["Codebook"])
        self.codebook.eval()
        
        self.transformer_config = {
            "vocab_size": self.codes,
            "block_size": self.channels,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**self.transformer_config).to(self.device)
        
        # how much to keep
        self.pkeep = 0.5 # random drop 30%

    # @staticmethod
    # def load_vqgan(args):
    #     model = VQGAN(args)
    #     model.load_checkpoint(args.checkpoint_path)
    #     model = model.eval()
    #     return model

    @torch.no_grad()
    def encode_to_z(self, x): # x: image
        im_enc = self.vqgan_encoder(x)
        #print("im_enc shape: ", im_enc.shape)
        enc_to_vq = self.prev_conv(im_enc)
        vq_out = self.codebook(enc_to_vq)
        quant_z = vq_out["embeddings"]
        indices = vq_out["encodings"]
        #quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return im_enc, quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices):
        #ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = F.embedding(indices, self.codebook.embeddings).reshape(indices.shape[0], self.latent_dim, self.latent_dim, self.latent_dim, self.channels)
        ix_to_vectors = self.postV_conv(ix_to_vectors.permute(0,4,1,2,3).contiguous())#.permute(0, 4, 1, 2, 3) #ix_to_vectors.permute(0, 3, 1, 2)
        assert ix_to_vectors.shape[1:] == (512,4,4,4), "something wrong with the shape"
        image = self.vqgan_decoder(ix_to_vectors)
        return ix_to_vectors, image

    def forward(self, x):
        # quant z is actually im_enc
        quant_z, _, indices = self.encode_to_z(x)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])
        #print("trans logits shape ", logits.shape)

        return quant_z, logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, top_k, top_p, temperature=1.0):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                #logits = self.top_k_logits(logits, top_k)
                logits = top_k_top_p_filtering(logits, top_k, top_p)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x


    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        with torch.no_grad():
            _, _, indices = self.encode_to_z(x)
            sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to(self.device)

            start_indices = indices[:, :indices.shape[1] // 2]
            sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1], top_k=self.codes//2, top_p=1)
            _,half_sample = self.z_to_image(sample_indices)

            start_indices = indices[:, :0]
            sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1], top_k=self.codes, top_p=1)
            _,full_sample = self.z_to_image(sample_indices)

            _,x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.cat((x, x_rec, half_sample, full_sample), dim = 0)

    @torch.no_grad()
    def sample_inf(self, x, c, steps, top_k, top_p, temperature=1.0):
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                #logits = self.top_k_logits(logits, top_k)
                logits = top_k_top_p_filtering(logits, top_k, top_p)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        #x = x[:, :-c.shape[1]]
        return x


class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.LayerNorm(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3
            
        return output


class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1,is_dis =True):
        super(Discriminator, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class # when is_dis = False, this is the encoder latent dim size
        
        self.conv1 = nn.Conv3d(1, channel//16, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(channel//16)
        self.conv2 = nn.Conv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//8)
        self.conv3 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//4)
        self.conv4 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel//2)
        self.conv5 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(channel)
        #self.conv6 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        self.l1 = nn.Linear(512 * 4 * 4 * 4, out_class)

        #self.post_linear = nn.Linear(n_class * 9, n_class)
        
    def forward(self, x):
        h1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        #h6 = self.conv6(h5)
        output = h5
        #print("finish all conv, now shape", output.shape)
        # if self.is_dis:
        #     output = self.l1(output.view(output.shape[0], -1))
        #     return output
        #else: # if is encoder
        output = self.l1(output.view(output.shape[0], -1))
            #output = self.post_linear(output)
        
        #print("is discriminator? {}, output discriminator size: {}".format(self.is_dis, output.shape))
        return output

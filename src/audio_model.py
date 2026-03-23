from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from audioclip import AudioCLIP
from utils.layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class GCN(nn.Module):
    def __init__(self, input_width, device):
        super().__init__()
        width = int(input_width / 2)
        self.gc1 = GraphConvolution(input_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(input_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj(device)
        self.linear = nn.Linear(input_width, input_width)
        self.gelu = QuickGELU()

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def forward(self, x, lengths):
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

class _NonLocalBlockND(nn.Module):#TSA
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]#判断输入参数是否满足条件

        self.dimension = dimension #1
        self.sub_sample = sub_sample #false

        self.in_channels = in_channels #512
        self.inter_channels = inter_channels #None

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2 #整除2，为256
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:#有batchNormalization层 true
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)#填充常数，即初始化为bn的weight与bias为0
            nn.init.constant_(self.W[1].bias, 0)
        else:#无batchNormalization层
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:#下采样 false
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, y, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        #x==[batch,512,T]
        batch_size = x.size(0) #batch
        #view按顺序进行维度变换，-1表示可以通过其他参数推断
        g_x = self.g(x).view(batch_size, self.inter_channels, -1) #[batch,256,T]
        g_x = g_x.permute(0, 2, 1) #[batch,T,256]
        #permute按维度进行维度变换
        theta_y = self.theta(y).view(batch_size, self.inter_channels, -1) #[batch,256,T]
        #theta_y = theta_y.permute(0, 2, 1) #[batch,T,256]
        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1) #[batch,256,T]
        phi_y = phi_y.permute(0,2,1)

        f = torch.matmul(g_x, theta_y)#tensor乘法[batch,T,T]
        N = f.size(-1) #T
        f_div_C = f / N #[batch,T,T]

        y = torch.matmul(f_div_C, phi_y) #[batch,T,256]
        y = y.permute(0, 2, 1).contiguous() #[batch,256,T]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])#[batch,256,T]
        W_y = self.W(y) #[batch,512,T]
        z = W_y + x #[batch,512,T]

        if return_nl_map:
            return z, f_div_C
        return z
class AudioCLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 embed_dim_audio: int,
                 visual_length: int,
                 visual_width: int,
                 audio_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.audio_width = audio_width
        self.embed_dim = embed_dim
        self.embed_dim_audio = embed_dim_audio
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        self.gcn = GCN(visual_width, device)
        self.gcn_audio = GCN(audio_width, device)

        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 2, visual_width))
        ]))

        self.mlp3 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(audio_width, audio_width * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(audio_width * 2, audio_width))
        ]))

        width = visual_width + audio_width
        self.classifier = nn.Linear(width, 1)

        self.cross_attn = _NonLocalBlockND(in_channels=visual_width, inter_channels=visual_width, dimension=1,
                                           sub_sample=False, bn_layer=True)
        self.cross_attn2 = _NonLocalBlockND(in_channels=audio_width, inter_channels=audio_width, dimension=1,
                                           sub_sample=False, bn_layer=True)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False
        self.clipmodel_audio, _ = clip.load("RN50", device)
        for clip_param in self.clipmodel_audio.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings_audio = nn.Embedding(77, self.embed_dim_audio)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)






        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings_audio.weight, std=0.01)
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask



    def encode_visual(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        x = self.gcn(x, lengths)

        return x

    def encode_audio(self, audio, lengths):
        x = audio.to(torch.float)
        x = self.gcn_audio(x, lengths)
        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def encode_textprompt_audio(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel_audio.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings_audio(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel_audio.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, images, audio, padding_mask, text, lengths):
        visual_features = self.encode_visual(images, padding_mask, lengths)
        audio_features = self.encode_audio(audio, lengths)
        text_features_ori_visual = self.encode_textprompt(text)
        text_features_ori_audio = self.encode_textprompt_audio(text)

        text_features_visual = text_features_ori_visual.unsqueeze(0)
        text_features_audio = text_features_ori_audio.unsqueeze(0)

        text_features_visual = text_features_visual.expand(visual_features.shape[0], text_features_visual.shape[1], text_features_visual.shape[2])
        text_features_visual = text_features_visual + self.mlp2(text_features_visual)

        text_features_audio = text_features_audio.expand(audio_features.shape[0], text_features_audio.shape[1], text_features_audio.shape[2])
        text_features_audio = text_features_audio + self.mlp3(text_features_audio)

        attn, visual_map = self.cross_attn(visual_features.permute(0,2,1), text_features_visual.permute(0,2,1), return_nl_map=True)
        attn = attn.permute(0,2,1)
        attn = attn / attn.norm(dim=-1, keepdim=True)
        visual_features = visual_features + attn

        attn2, audio_map = self.cross_attn2(audio_features.permute(0,2,1), text_features_audio.permute(0,2,1), return_nl_map=True)
        attn2 = attn2.permute(0,2,1)
        attn2 = attn2 / attn2.norm(dim=-1, keepdim=True)
        audio_features = audio_features + attn2

        features = torch.cat((visual_features, audio_features), dim=-1)
        logits1 = self.classifier(features)

        logits2 = (visual_map + audio_map)/0.14

        return text_features_ori_visual, text_features_ori_audio, logits1, logits2



class AudioCLIPVAD_OnlyVideo(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 embed_dim_audio: int,
                 visual_length: int,
                 visual_width: int,
                 audio_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.audio_width = audio_width
        self.embed_dim = embed_dim
        self.embed_dim_audio = embed_dim_audio
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        self.gcn = GCN(visual_width, device)

        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 2, visual_width))
        ]))

        self.classifier = nn.Linear(visual_width, 1)

        self.cross_attn = _NonLocalBlockND(in_channels=visual_width, inter_channels=visual_width, dimension=1,
                                           sub_sample=False, bn_layer=True)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask



    def encode_visual(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        x = self.gcn(x, lengths)

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, images, audio, padding_mask, text, lengths):
        #visual_features = self.encode_visual(images, padding_mask, lengths)
        visual_features = images.to(torch.float).to(self.device) + 1e-8
        text_features_ori_visual = self.encode_textprompt(text)

        text_features_visual = text_features_ori_visual.unsqueeze(0)

        text_features_visual = text_features_visual.expand(visual_features.shape[0], text_features_visual.shape[1], text_features_visual.shape[2])
        #text_features_visual = text_features_visual + self.mlp2(text_features_visual)

        visual_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_norm = text_features_visual/text_features_visual.norm(dim=-1, keepdim=True)
        text_norm = text_norm.permute(0,2,1)
        visual_map = visual_norm @ text_norm

        #attn, visual_map = self.cross_attn(visual_features.permute(0,2,1), text_features_visual.permute(0,2,1), return_nl_map=True)
        #attn = attn.permute(0,2,1)
        #attn = attn / attn.norm(dim=-1, keepdim=True)
        #visual_features = visual_features + attn

        features = visual_features
        logits1 = self.classifier(features)

        logits2 = visual_map/0.07


        return text_features_ori_visual, text_features_ori_visual, logits1, logits2

class AudioCLIPVAD_OnlyAudio(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 embed_dim_audio: int,
                 visual_length: int,
                 visual_width: int,
                 audio_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.audio_width = audio_width
        self.embed_dim = embed_dim
        self.embed_dim_audio = embed_dim_audio
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.gcn_audio = GCN(audio_width, device)

        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 2, visual_width))
        ]))

        self.mlp3 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(audio_width, audio_width * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(audio_width * 2, audio_width))
        ]))

        self.classifier = nn.Linear(audio_width, 1)

        self.cross_attn2 = _NonLocalBlockND(in_channels=audio_width, inter_channels=audio_width, dimension=1,
                                           sub_sample=False, bn_layer=True)

        self.clipmodel_audio, _ = clip.load("RN50", device)
        for clip_param in self.clipmodel_audio.parameters():
            clip_param.requires_grad = False

        self.text_prompt_embeddings_audio = nn.Embedding(77, self.embed_dim_audio)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings_audio.weight, std=0.01)

    def encode_audio(self, audio, lengths):
        x = audio.to(torch.float)
        x = self.gcn_audio(x, lengths)
        return x

    def encode_textprompt_audio(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel_audio.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings_audio(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel_audio.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, images, audio, padding_mask, text, lengths):
        #audio_features = self.encode_audio(audio, lengths)
        audio_features = audio.to(torch.float).to(self.device) + 1e-8
        text_features_ori_audio = self.encode_textprompt_audio(text)

        text_features_audio = text_features_ori_audio.unsqueeze(0)

        text_features_audio = text_features_audio.expand(audio_features.shape[0], text_features_audio.shape[1], text_features_audio.shape[2])
        #text_features_audio = text_features_audio + self.mlp3(text_features_audio)

        audio_norm = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_norm = text_features_audio / text_features_audio.norm(dim=-1, keepdim=True)
        text_norm = text_norm.permute(0, 2, 1)
        audio_map = audio_norm @ text_norm

        #attn2, audio_map = self.cross_attn2(audio_features.permute(0,2,1), text_features_audio.permute(0,2,1), return_nl_map=True)
        #attn2 = attn2.permute(0,2,1)
        #attn2 = attn2 / attn2.norm(dim=-1, keepdim=True)
        #audio_features = audio_features + attn2

        features = audio_features
        logits1 = self.classifier(features)

        #logits2 = (visual_map + audio_map)/0.14
        logits2 = audio_map/0.07


        return text_features_ori_audio, text_features_ori_audio, logits1, logits2
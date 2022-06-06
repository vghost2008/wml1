import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape)==4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = self.weight[:, None] * x + self.bias[:, None]
            return x

class SEBlock(nn.Module):
    def __init__(self,channels,r=16):
        super().__init__()
        self.channels = channels
        self.r = r
        self.fc0 = nn.Linear(self.channels,self.channels//r)
        self.fc1 = nn.Linear(self.channels//r,self.channels)

    def forward(self,net):
        org_net = net
        net = net.mean(dim=(2,3),keepdim=False)
        net = self.fc0(net)
        net = F.relu(net,inplace=True)
        net = self.fc1(net)
        net = F.sigmoid(net)
        net = torch.unsqueeze(net,dim=-1)
        net = torch.unsqueeze(net,dim=-1)
        return net*org_net

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256,max_rows=50,max_cols=50):
        super().__init__()
        self.row_embed = nn.Embedding(max_rows, num_pos_feats)
        self.col_embed = nn.Embedding(max_cols, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        '''

        Args:
            x: [...,C,H,W]
        Returns:

        '''
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
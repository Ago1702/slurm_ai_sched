import torch
import torch.nn as nn
from torch.distributions import Normal

MAX_INT = 2**16

class TransformerLayer(nn.Module):

    def __init__(self, embedded_dim:int=128, num_heads:int=4, dropout:float=0.3, batch_first:bool=True):
        super().__init__()

        self.embedded_dim = embedded_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.projection = nn.Linear(embedded_dim, embedded_dim * 3)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedded_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        self.attention_norm = nn.LayerNorm(embedded_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embedded_dim, embedded_dim),
            nn.LayerNorm(embedded_dim),
            nn.ReLU(),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        q, k, v = torch.chunk(self.projection.forward(x), 3, dim=-1)
        x_att, _ = self.attention_layer.forward(q, k, v)
        x_att = self.attention_norm(x_att)
        x = x + x_att
        x_ffwd = self.ffwd(x)
        return x + x_ffwd

class QNet(nn.Module):
    '''
    Class QNet: First implementation of Critic Net
    '''

    def __init__(self, input_dim:int, output_dim:int, depth:int=5, embedded_dim:int=128, num_heads:int=4, batch_first:bool=True):
        super(QNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedded_dim = embedded_dim
        self.num_heads = num_heads
        self.depth = depth
        self.batch_first = batch_first
        self.cls = nn.Parameter(torch.randn(size=(1, 1, self.embedded_dim)))

        self.dimensioner = nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, self.embedded_dim))
        self.attentions = nn.ModuleList([TransformerLayer(
            embedded_dim=self.embedded_dim,
            num_heads=self.num_heads,
            dropout=0.3,
            batch_first=self.batch_first
        ) for _ in range(self.depth)])
        self.classifier = nn.Linear(self.embedded_dim, self.output_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x = self.dimensioner(x)
        x = torch.cat([torch.tile(self.cls, (N, 1, 1)), x], dim=1)
        for attention in self.attentions:
            x = attention.forward(x)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)
        
class ActionNet(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int=128, depth:int=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.eps = 0.99
        self.input_layer = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        ) for _ in range(depth)])
        self.mean = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        self.std = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self.input_layer(x)
        for hidden in self.hidden:
            x_h = hidden(x)
            x = x + x_h
        mean = self.mean(x)
        log_std = self.std(x)
        value = Normal(mean, log_std.exp()).rsample()
        value = (1 - self.eps) * value + self.eps * torch.rand_like(value) * 3600 * 24
        return value, mean, log_std
        
if __name__ == '__main__':
    qnet = QNet(input_dim=5, output_dim=1)
    data = torch.rand((4, 10, 5))
    #print(qnet)
    res = qnet.forward(data)
    print(data.shape)
    print(res.shape)
    a = ActionNet(5, 1)
    print(a.forward(data).shape)
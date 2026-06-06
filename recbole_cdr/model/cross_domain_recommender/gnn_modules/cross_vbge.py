import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN, kld_gauss


class CrossDGCNLayer(nn.Module):
    def __init__(self, opt):
        super(CrossDGCNLayer, self).__init__()
        self.gc1 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc2 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc3 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.src_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.tgt_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        rate = torch.tensor(opt["rate"]).view(-1)
        self.register_buffer("rate", rate)

    def forward(self, src_u, tgt_u, src_UV, src_VU, tgt_UV, tgt_VU):
        src_ho = self.gc3(self.gc1(src_u, src_VU), src_UV)
        tgt_ho = self.gc4(self.gc2(tgt_u, tgt_VU), tgt_UV)
        src_out = F.relu(self.src_union(torch.cat((src_ho, src_u), dim=1)))
        tgt_out = F.relu(self.tgt_union(torch.cat((tgt_ho, tgt_u), dim=1)))
        mixed = self.rate * src_out + (1 - self.rate) * tgt_out
        return mixed, mixed


class CrossLastLayer(nn.Module):
    def __init__(self, opt):
        super(CrossLastLayer, self).__init__()
        self.opt = opt
        self.gc1 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc2 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc3_mean = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc3_logstd = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4_mean = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4_logstd = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.src_mean = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.src_logstd = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.tgt_mean = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.tgt_logstd = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        rate = torch.tensor(opt["rate"]).view(-1)
        self.register_buffer("rate", rate)

    def forward(self, src_u, tgt_u, src_UV, src_VU, tgt_UV, tgt_VU):
        src_ho = self.gc1(src_u, src_VU)
        tgt_ho = self.gc2(tgt_u, tgt_VU)
        src_mean = self.src_mean(torch.cat((self.gc3_mean(src_ho, src_UV), src_u), dim=1))
        src_logstd = self.src_logstd(torch.cat((self.gc3_logstd(src_ho, src_UV), src_u), dim=1))
        tgt_mean = self.tgt_mean(torch.cat((self.gc4_mean(tgt_ho, tgt_UV), tgt_u), dim=1))
        tgt_logstd = self.tgt_logstd(torch.cat((self.gc4_logstd(tgt_ho, tgt_UV), tgt_u), dim=1))
        mean = self.rate * src_mean + (1 - self.rate) * tgt_mean
        logstd = self.rate * src_logstd + (1 - self.rate) * tgt_logstd
        return mean, logstd


class CrossVBGE(nn.Module):
    def __init__(self, opt):
        super(CrossVBGE, self).__init__()
        self.dropout = opt["dropout"]
        layers = [CrossDGCNLayer(opt) for _ in range(opt["GNN"] - 1)] + [CrossLastLayer(opt)]
        self.encoder = nn.ModuleList(layers)

    def forward(self, src_u, tgt_u, src_UV, src_VU, tgt_UV, tgt_VU):
        s, t = src_u, tgt_u
        for layer in self.encoder[:-1]:
            s = F.dropout(s, self.dropout, training=self.training)
            t = F.dropout(t, self.dropout, training=self.training)
            s, t = layer(s, t, src_UV, src_VU, tgt_UV, tgt_VU)
        mean, logstd = self.encoder[-1](s, t, src_UV, src_VU, tgt_UV, tgt_VU)
        return mean, logstd

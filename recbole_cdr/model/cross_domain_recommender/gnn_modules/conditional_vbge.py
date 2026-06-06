import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN, kld_gauss


class CondiDGCNLayer(nn.Module):
    def __init__(self, opt):
        super(CondiDGCNLayer, self).__init__()
        self.gc1 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc2 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc3 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.src_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.tgt_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])

    def forward(self, src_u, tgt_u, src_UV, src_VU, tgt_UV, tgt_VU):
        src_ho = self.gc3(self.gc1(src_u, src_VU), src_UV)
        tgt_ho = self.gc4(self.gc2(tgt_u, tgt_VU), tgt_UV)
        src_out = F.relu(self.src_union(torch.cat((src_ho, src_u), dim=1)))
        tgt_out = F.relu(self.tgt_union(torch.cat((tgt_ho, tgt_u), dim=1)))
        return src_out, tgt_out


class CondiLastLayer(nn.Module):
    def __init__(self, opt):
        super(CondiLastLayer, self).__init__()
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

    def _reparameters(self, mean, logstd):
        logstd = torch.clamp(logstd, -10, 10)
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        if self.training:
            noise = torch.randn(mean.size(0), mean.size(1)).to(mean.device)
            sampled_z = noise * sigma + mean
        else:
            sampled_z = mean
        kld = kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld

    def forward(self, src_u, tgt_u, src_UV, src_VU, tgt_UV, tgt_VU):
        src_ho = self.gc1(src_u, src_VU)
        tgt_ho = self.gc2(tgt_u, tgt_VU)
        src_mean = self.src_mean(torch.cat((self.gc3_mean(src_ho, src_UV), src_u), dim=1))
        src_logstd = self.src_logstd(torch.cat((self.gc3_logstd(src_ho, src_UV), src_u), dim=1))
        tgt_mean = self.tgt_mean(torch.cat((self.gc4_mean(tgt_ho, tgt_UV), tgt_u), dim=1))
        tgt_logstd = self.tgt_logstd(torch.cat((self.gc4_logstd(tgt_ho, tgt_UV), tgt_u), dim=1))
        return src_mean, src_logstd, tgt_mean, tgt_logstd


class ConditionalVBGE(nn.Module):
    def __init__(self, opt):
        super(ConditionalVBGE, self).__init__()
        self.dropout = opt["dropout"]
        self.is_conditional = opt.get("isConditional", True)
        self.is_condi_norm = opt.get("isCondi_norm", False)
        self.condi_src = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.condi_tgt = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        layers = [CondiDGCNLayer(opt) for _ in range(opt["GNN"] - 1)] + [CondiLastLayer(opt)]
        self.encoder = nn.ModuleList(layers)

    def _feature_normalize(self, x):
        rowsum = torch.div(1.0, x.sum(dim=1))
        rowsum[torch.isinf(rowsum)] = 0.
        return torch.mm(torch.diag(rowsum), x)

    def _conditional(self, s, s_condi, t, t_condi):
        src = torch.cat((s, t_condi), dim=1)
        tgt = torch.cat((t, s_condi), dim=1)
        if self.is_condi_norm:
            return self._feature_normalize(F.relu(self.condi_src(src))), \
                   self._feature_normalize(F.relu(self.condi_tgt(tgt)))
        return F.relu(self.condi_src(src)), F.relu(self.condi_tgt(tgt))

    def forward(self, src_u, tgt_u, src_UV, src_VU, tgt_UV, tgt_VU, condi_src, condi_tgt):
        s = src_u if not self.is_conditional else src_u
        t = tgt_u if not self.is_conditional else tgt_u
        for layer in self.encoder[:-1]:
            if self.is_conditional:
                s, t = self._conditional(src_u, condi_src, tgt_u, condi_tgt)
            s = F.dropout(s, self.dropout, training=self.training)
            t = F.dropout(t, self.dropout, training=self.training)
            s, t = layer(s, t, src_UV, src_VU, tgt_UV, tgt_VU)
        if self.is_conditional:
            s, t = self._conditional(src_u, condi_src, tgt_u, condi_tgt)
        return self.encoder[-1](s, t, src_UV, src_VU, tgt_UV, tgt_VU)

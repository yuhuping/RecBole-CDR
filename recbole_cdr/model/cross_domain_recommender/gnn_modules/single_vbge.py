import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN, kld_gauss


class DGCNLayer(nn.Module):
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.gc1 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc2 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc3 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4 = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.user_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc3(self.gc1(ufea, VU_adj), UV_adj)
        Item_ho = self.gc4(self.gc2(vfea, UV_adj), VU_adj)
        User = F.relu(self.user_union(torch.cat((User_ho, ufea), dim=1)))
        Item = F.relu(self.item_union(torch.cat((Item_ho, vfea), dim=1)))
        return User, Item

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc3(self.gc1(ufea, VU_adj), UV_adj)
        return F.relu(self.user_union(torch.cat((User_ho, ufea), dim=1)))


class LastLayer(nn.Module):
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt = opt
        self.gc1 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc2 = GCN(opt["feature_dim"], opt["hidden_dim"], opt["dropout"], opt["leakey"])
        self.gc3_mean = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc3_logstd = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4_mean = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.gc4_logstd = GCN(opt["hidden_dim"], opt["feature_dim"], opt["dropout"], opt["leakey"])
        self.user_union_mean = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.user_union_logstd = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.item_union_mean = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])
        self.item_union_logstd = nn.Linear(opt["feature_dim"] * 2, opt["feature_dim"])

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

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        user, u_kld = self._forward_user(ufea, vfea, UV_adj, VU_adj)
        item, i_kld = self._forward_item(ufea, vfea, UV_adj, VU_adj)
        self.kld_loss = u_kld + i_kld
        return user, item

    def _forward_user(self, ufea, vfea, UV_adj, VU_adj):
        ho = self.gc1(ufea, VU_adj)
        mean = self.user_union_mean(torch.cat((self.gc3_mean(ho, UV_adj), ufea), dim=1))
        logstd = self.user_union_logstd(torch.cat((self.gc3_logstd(ho, UV_adj), ufea), dim=1))
        return self._reparameters(mean, logstd)

    def _forward_item(self, ufea, vfea, UV_adj, VU_adj):
        ho = self.gc2(vfea, UV_adj)
        mean = self.item_union_mean(torch.cat((self.gc4_mean(ho, VU_adj), vfea), dim=1))
        logstd = self.item_union_logstd(torch.cat((self.gc4_logstd(ho, VU_adj), vfea), dim=1))
        return self._reparameters(mean, logstd)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        ho = self.gc1(ufea, VU_adj)
        mean = self.user_union_mean(torch.cat((self.gc3_mean(ho, UV_adj), ufea), dim=1))
        logstd = self.user_union_logstd(torch.cat((self.gc3_logstd(ho, UV_adj), ufea), dim=1))
        return mean, logstd


class SingleVBGE(nn.Module):
    def __init__(self, opt):
        super(SingleVBGE, self).__init__()
        self.dropout = opt["dropout"]
        layers = [DGCNLayer(opt) for _ in range(opt["GNN"] - 1)] + [LastLayer(opt)]
        self.encoder = nn.ModuleList(layers)

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        u, v = ufea, vfea
        for layer in self.encoder:
            u = F.dropout(u, self.dropout, training=self.training)
            v = F.dropout(v, self.dropout, training=self.training)
            u, v = layer(u, v, UV_adj, VU_adj)
        return u, v

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        u = ufea
        for layer in self.encoder[:-1]:
            u = F.dropout(u, self.dropout, training=self.training)
            u = layer.forward_user_share(u, UV_adj, VU_adj)
        return self.encoder[-1].forward_user_share(u, UV_adj, VU_adj)

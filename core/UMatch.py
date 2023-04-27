import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import batch_episym


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.InstanceNorm1d(channels, eps=1e-3),
                nn.SyncBatchNorm(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm1d(out_channels, eps=1e-3),
                nn.SyncBatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=1),
            )
            
    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.SyncBatchNorm(2*channels), nn.ReLU(inplace=True),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2, weight_v = None):
        #motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        if weight_v is not None:
            value = value * weight_v.view(batch_size, 1, 1, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new

class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Conv1d(4, channels, kernel_size=1)
    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)

        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, 1, kernel_size=1)
        )
                                                                
    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)
    
class GlobalContext(nn.Module):
    def __init__(self, channels, reduction=4):
        super(GlobalContext, self).__init__()
        inter_channels = int(channels // reduction)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.SyncBatchNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(2*channels, channels, kernel_size=1),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
               
    def forward(self, x, logits):
        w = logits.unsqueeze(1) # b*1*n
        w = torch.tanh(torch.relu(w))
        w = F.normalize(w, p=1, dim=2)
        
        x_w = torch.mul(x, w.expand_as(x)) # b*c*n
        x_sum =torch.sum(x_w,dim=2, keepdim=True) # bc1
        
        global_context = F.normalize(self.conv(x_sum), p=2, dim=1) # bc1
        
        proj_length = torch.bmm(x.transpose(1,2), global_context).transpose(1,2) # b1n
        proj = torch.mul(proj_length, global_context) # bcn
        orth_comp = x - proj # bcn
        final_feat = x + self.conv1(torch.cat([orth_comp, global_context.expand_as(orth_comp)], dim=1)) #b 2c n => b c n

        return final_feat
    

class Pool(nn.Module):
    def __init__(self, channels, head, k, drop_p=0.):
        nn.Module.__init__(self)
        self.k = k

        self.init_filter = PointCN(channels)
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0 else nn.Identity()
        self.proj = nn.Linear(channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.pool = AttentionPropagation(channels, head)

    def top_k_graph(self, scores, x, k):
        # x: BCN
        x = self.init_filter(x)
        num_nodes = x.shape[-1]
        num_sampled_nodes = int(k * num_nodes)
        values, idx = torch.topk(scores, num_sampled_nodes, dim=-1)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            values = values.unsqueeze(0)
        idx_gather = idx.unsqueeze(1).repeat(1, x.shape[1], 1) # BK->BCK
        x_new = torch.gather(x, 2, idx_gather) # BCK
        values = values.unsqueeze(1)
        x_new = torch.mul(x_new, values.repeat(1, x.shape[1], 1))
        return x_new, idx

    def forward(self, x):
        # x: BCN
        Z = self.drop(x).permute(0,2,1)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        x_new, idx = self.top_k_graph(scores, x, self.k) # x_new: BCK
        x_new = self.pool(x_new, x) # BCK
        return x_new


class Unpool(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.unpool = AttentionPropagation(channels, head)

    def forward(self, x, x_ori):
        # x: BCK, x_ori: BCN
        x_new = self.unpool(x_ori, x) # BCN
        return x_new


class BottomGAT(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.gat = AttentionPropagation(channels, head)

    def forward(self, d):
        d = self.gat(d, d)
        return d


class GUFBlock(nn.Module):
    def __init__(self, channels, head, ks, drop_p=0.):
        nn.Module.__init__(self)
        self.ks = ks
        self.pools = nn.ModuleList()
        self.bottom_gat = BottomGAT(channels, head)
        self.unpools = nn.ModuleList()
        for k in self.ks:
            self.pools.append(Pool(channels, head, k))
            self.unpools.append(Unpool(channels, head))
        self.inlier_pre = InlinerPredictor(channels)

    def forward(self, xs, d):
        # xs: B1N4, d: BCN
        d_ori_list = []
        for i in range(len(self.ks)):
            d_ori_list.append(d)
            d = self.pools[i](d)
        d = self.bottom_gat(d)
        d_ori_list = d_ori_list[::-1]
        for j in range(len(self.ks)):
            d = self.unpools[j](d, d_ori_list[j])
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre(d), 1)
        e_hat = weighted_8points(xs, logits)
        return d, logits, e_hat


class UMatch(nn.Module):
    def __init__(self, config, use_gpu=True):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num
        self.init_project = InitProject(config.net_channels)
        self.layer_blocks = nn.Sequential(
            *[GUFBlock(config.net_channels, config.head, config.r_list) for _ in range(self.layer_num)]
        )
        self.global_context1 = GlobalContext(config.net_channels)
        self.global_context2 = GlobalContext(config.net_channels)
     
    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1N4 -> B4N
        input = data['xs'].transpose(1,3).squeeze(3)
        d = self.init_project(input)# BCN

        res_logits, res_e_hat = [], []
        for i in range(self.layer_num):
            d, logits, e_hat = self.layer_blocks[i](data['xs'], d) # BCN
            res_logits.append(logits), res_e_hat.append(e_hat)
            if i == 1:
                d = self.global_context1(d, logits)
            if i == 3:
                d = self.global_context2(d, logits)

        return res_logits, res_e_hat 


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


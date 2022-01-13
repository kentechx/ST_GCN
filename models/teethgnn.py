import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_dgcnn import STN, Backbone_Seg, Backbone_Cls, SharedMLP1d, EdgeConv, knn


class GFU(nn.Module):
    """
    Gated feature unit.
    """
    def __init__(self, x_channels, h_channels):
        super(GFU, self).__init__()
        self.wxr = nn.Conv1d(x_channels, h_channels, kernel_size=1, bias=True)
        self.wxz = nn.Conv1d(x_channels, h_channels, kernel_size=1, bias=True)
        self.wxn = nn.Conv1d(x_channels, h_channels, kernel_size=1, bias=True)

        self.whr = nn.Conv1d(h_channels, h_channels, kernel_size=1, bias=True)
        self.whz = nn.Conv1d(h_channels, h_channels, kernel_size=1, bias=True)
        self.whn = nn.Conv1d(h_channels, h_channels, kernel_size=1, bias=True)

    def forward(self, x, h):
        r = torch.sigmoid(self.wxr(x)+self.whr(h))    # (batch, k, n_p)
        z = torch.sigmoid(self.wxz(x)+self.whz(h))    # (batch, k, n_p)
        n = nn.LeakyReLU(0.2)(self.wxn(x)+ r*self.whn(h))     # (N, 64, n_p)
        return (1-z) * n + z * h


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dk=64):
        super(AttentionLayer, self).__init__()
        self.wq = nn.Conv1d(in_channels, dk, kernel_size=1, bias=False)
        self.wk = nn.Conv1d(in_channels, dk, kernel_size=1, bias=False)
        self.wv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.wq(x).transpose(1, 2)      # (batch, n, dk)
        dk = q.shape[2]
        k = self.wk(x)      # (batch, dk, n)
        v = self.wv(x)      # (batch, out_channels, n)
        return v@(self.softmax(q@k/(dk**0.5)))


class OffsetBranch(nn.Module):
    def __init__(self, args):
        super(OffsetBranch, self).__init__()

        self.smlp1 = SharedMLP1d([args.emb_dims+args.n_edgeconvs_backbone*64, 256, 128], args.norm)
        self.dp = nn.Dropout2d(args.dropout)
        self.smlp2 = nn.Conv1d(128, 3, kernel_size=1)

    def forward(self, x):
        return self.smlp2(self.dp(self.smlp1(x)))


class SemanticBranch(nn.Module):
    def __init__(self, args):
        super(SemanticBranch, self).__init__()
        self.k = args.k

        self.smlp1 = SharedMLP1d([args.emb_dims+args.n_edgeconvs_backbone*64, 256], args.norm)
        self.conv = EdgeConv([256*2, 256], args.k, args.norm)
        self.smlp2 = nn.Sequential(SharedMLP1d([256, 256], args.norm),
                                   nn.Dropout2d(args.dropout),
                                   SharedMLP1d([256, 128], args.norm),
                                   nn.Conv1d(128, args.output_channels, kernel_size=1))

    def forward(self, x, p):
        x = self.smlp1(x)
        idx = knn(p, self.k)
        x = self.conv(x, idx)
        return self.smlp2(x)


class TeethGNN(nn.Module):
    """
    @param args: k, number of neighbors
                input_channels, int
                output_channels, int
                dynamic, bool, if using dynamic or not.
                transform, bool, if using the transform net
                n_edgeconvs_backbone, int, the number of EdgeConvs in the backbone
                emb_dims, int
                global_pool_backbone, str, "avg" or "max"
                norm, str, "instance" or "batch"
                dropout, float
    """
    def __init__(self, args):
        super(TeethGNN, self).__init__()

        if args.use_stn:
            self.stn = STN(args.k, args.norm)

        self.backbone = Backbone_Seg(args)
        self.offset_branch = OffsetBranch(args)
        self.semantic_branch = SemanticBranch(args)

    def forward(self, x):
        device = x.device
        p = x[:, :3, :].contiguous()
        if hasattr(self, "stn"):
            if not hasattr(self, 'c'):
                self.c = torch.zeros((x.shape[0], 15, 15), dtype=torch.float32, device=device)
                for i in range(0, 15, 3):
                    self.c[:, i:i+3, i:i+3] = 1

            t = self.stn(x[:, :3, :].contiguous())
            t = t.repeat(1, 5, 5)       # (batch_size, 15, 15)
            t1 = self.c * t
            x = torch.bmm(t1, x)
        else:
            t = torch.ones((1, 1), device=device)

        x = self.backbone(x)

        offsets = self.offset_branch(x)
        p = p + offsets.detach()    # (n, 3, 10000)

        x = self.semantic_branch(x, p)

        return x, t, offsets

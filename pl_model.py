import torch, random, numpy as np, torch.nn.functional as F, torch.nn as nn
from torch.utils.data import DataLoader
from models.my_dgcnn import MyDGCNN_Seg
# from data.dental import DentalSegDataset
from data.dental_sample import DentalSegDataset
from data.common import calc_features

import pytorch_lightning as pl
import torchmetrics


class LitTeethGNN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.net = MyDGCNN_Seg(args)
        self.train_iou = torchmetrics.IoU(17, ignore_index=0)
        self.val_iou = torchmetrics.IoU(17, ignore_index=0)
        self.test_iou = torchmetrics.IoU(17, ignore_index=0)

    def forward(self, x):
        return self.net(x)

    def infer(self, vs, ts):
        vs = vs - vs.mean(0)  # preprocess
        X = calc_features(vs, ts)[None]         # (1, 15, 1024)
        pred, trans, offsets = self.forward(X)
        return pred[0].softmax(0), trans, offsets[0]

    def training_step(self, batch, batch_idx):
        delta = self.hparams.args.delta
        x, y, y_offsets, fp = batch
        out, t, out_offsets = self(x)
        ce = F.cross_entropy(out, y)
        l2 = torch.norm(out_offsets-y_offsets, dim=1).mean()
        loss = ce + delta * l2
        self.log('loss/ce', ce, True)
        self.log('loss/l2', l2, True)
        self.log('loss', loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        # self.log('iou', self.train_iou(F.softmax(out, 1), y), True)
        return loss

    def validation_step(self, batch, batch_idx):
        delta = self.hparams.args.delta
        x, y, y_offsets, fp = batch
        out, t, out_offsets = self(x)
        ce = F.cross_entropy(out, y)
        l2 = torch.norm(out_offsets-y_offsets, dim=1).mean()
        loss = ce + delta * l2
        self.log('val_loss/ce', ce, True)
        self.log('val_loss/l2', l2, True)
        self.log('val_loss', loss, True)

        self.val_iou(F.softmax(out, 1), y)
        self.log('val_iou', self.val_iou, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        delta = self.hparams.args.delta
        x, y, y_offsets, fp = batch
        out, t, out_offsets = self(x)
        ce = F.cross_entropy(out, y)
        l2 = torch.norm(out_offsets-y_offsets, dim=1).mean()
        loss = ce + delta * l2
        self.log('test_loss/ce', ce, True)
        self.log('test_loss/l2', l2, True)
        self.log('test_loss', loss, True)

        self.test_iou(F.softmax(out, 1), y)
        self.log('test_iou', self.test_iou, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs, steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(DentalSegDataset(args, args.train_folder, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(DentalSegDataset(args, args.val_folder, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(DentalSegDataset(args, args.test_folder, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)

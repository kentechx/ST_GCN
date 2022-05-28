import time

import torch, torch.nn.functional as F, torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
# from data.gum_graph import GumSegDataset
from data.st_face_data import ST_FaceData
from data.common import calc_features

from torch_geometric.data import Data
# from torch_geometric.nn.models import GraphUNet
from models.deepergcn import DeeperGCN


def get_parser():
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/st_data2_face_gcn.yaml", help="path to config file")
    parser.add_argument('--gpus', type=int, default=1)

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.net = DeeperGCN(args.i_node_channels, args.i_edge_channels, args.o_channels, args.hidden_channels, args.num_layers)
        self.train_iou = torchmetrics.IoU(args.o_channels)
        self.val_iou = torchmetrics.IoU(args.o_channels)
        self.test_iou = torchmetrics.IoU(args.o_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr):
        return self.net(x, edge_index, edge_attr)

    def infer(self, vs, ts):
        vs = vs - vs.mean(0)  # preprocess
        X = calc_features(vs, ts)[None]         # (1, 15, 1024)
        pred, trans = self.forward(X, None)
        return pred[0].softmax(0), trans

    def training_step(self, batch_x, batch_idx):
        X, y, edge_index, edge_attr, fps = batch_x
        out = self(X, edge_index, edge_attr)
        loss = F.cross_entropy(out, y)
        self.log('loss', loss, batch_size=self.hparams.args.batch_size)
        self.log('lr', self.optimizers().param_groups[0]['lr'], batch_size=self.hparams.args.batch_size)
        # self.log('iou', self.train_iou(F.softmax(out, 1), y), True)
        return loss

    def validation_step(self, batch_x, batch_idx):
        X, y, edge_index, edge_attr, fps = batch_x
        out = self(X, edge_index, edge_attr)
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, True, batch_size=self.hparams.args.batch_size)

        self.val_iou(F.softmax(out, 1), y)
        self.log('val_iou', self.val_iou, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.args.batch_size)

    def test_step(self, batch_x, batch_idx):
        X, y, edge_index, edge_attr, fps = batch_x
        out = self(X, edge_index, edge_attr)
        loss = F.cross_entropy(out, y)
        self.log('test_loss', loss, True, batch_size=self.hparams.args.batch_size)

        self.test_iou(F.softmax(out, 1), y)
        self.log('test_iou', self.test_iou, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.args.batch_size)

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
        return DataLoader(ST_FaceData(args, args.train_folder, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=False,
                          collate_fn=ST_FaceData.collate)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(ST_FaceData(args, args.val_folder, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=False,
                          collate_fn=ST_FaceData.collate)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(ST_FaceData(args, args.test_folder, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=False,
                          collate_fn=ST_FaceData.collate)


if __name__ == "__main__":
    # from torch.multiprocessing import set_start_method
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass

    args = get_parser()
    pl.seed_everything(args.seed)

    model = LitModel(args)
    if args.load_from_checkpoint:
        model = LitModel.load_from_checkpoint(args.load_from_checkpoint)

    logger = TestTubeLogger("runs", args.experiment)
    callback = ModelCheckpoint(monitor='val_loss', save_top_k=20, save_last=True, mode='min')

    debug = False
    debug_args = {'limit_train_batches': 10} if debug else {}
    trainer = pl.Trainer(logger, gpus=args.gpus, max_epochs=args.max_epochs, progress_bar_refresh_rate=10, callbacks=[callback],
                         resume_from_checkpoint=args.resume_from_checkpoint, **debug_args)
    trainer.fit(model)

    results = trainer.test()
    print(results)
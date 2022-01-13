from pl_model import LitTeethGNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def get_parser():
    import argparse, yaml
    parser = argparse.ArgumentParser(description='Tooth Classification')
    parser.add_argument("--config", type=str, default="config/teethgnn_dental20000.yaml", help="path to config file")

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


if __name__ == "__main__":
    args = get_parser()
    pl.seed_everything(args.seed)

    model = LitTeethGNN(args)
    if args.load_from_checkpoint:
        model = LitTeethGNN.load_from_checkpoint(args.load_from_checkpoint)

    logger = TestTubeLogger("runs", args.experiment)
    callback = ModelCheckpoint(monitor='val_loss', save_top_k=20, save_last=True, mode='min')

    debug = False
    debug_args = {'limit_train_batches': 10} if debug else {}
    trainer = pl.Trainer(logger, gpus=1, max_epochs=args.max_epochs, progress_bar_refresh_rate=10, callbacks=[callback],
                         resume_from_checkpoint=args.resume_from_checkpoint, **debug_args)
    trainer.fit(model)

    results = trainer.test()
    print(results)
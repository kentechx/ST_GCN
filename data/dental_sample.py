import glob, os, numpy as np
import torch.utils.data
import torch
from .common import calc_features


class DentalSegDataset(torch.utils.data.Dataset):
    """
    @param k: k neighbors
    """

    def __init__(self, args, folder:str, train:bool):
        # args
        self.root_dir = args.root_dir           # the root directory of features
        self.augmentation = args.augmentation   # true or false
        self.num_points = args.num_points
        if not train:
            self.augmentation = False

        self.train = train

        # fps
        fps = glob.glob(os.path.join(self.root_dir, folder, "*"+args.suffix))
        fps = [os.path.abspath(fp) for fp in fps]
        fps = sorted(fps)
        self.fps = fps

        # if read all
        self.data = self.read_data(fps)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        """
        Get a single batch
        @return X, y: X with shape(10000, n_features), y with shape(10000, )
        """
        fp = self.fps[idx]
        vs, ts, offsets, y = torch.load(fp)
        vs -= vs.mean(0)        # preprocess
        if self.train:
            if self.augmentation:
                vs, ts, offsets, y = self.augment(vs, ts, offsets, y)
            # sample
            idx = np.random.permutation(len(ts))[:self.num_points]
            ts = ts[idx]
            y = y[idx]
            offsets = offsets[idx].T
        else:
            np.random.seed(111)
            idx = np.random.permutation(len(ts))[:self.num_points]
            ts, y = ts[idx], y[idx]
            offsets = offsets[idx].T

        vs = torch.tensor(vs, dtype=torch.float32)
        ts = torch.tensor(ts, dtype=torch.long)
        offsets = torch.tensor(offsets, dtype=torch.float32)
        X = calc_features(vs, ts)
        y = torch.tensor(y, dtype=torch.long)

        return X, y, offsets, fp

    def augment(self, vs, ts, offsets, ls):
        # jitter vertices
        sigma, clip = 0.01, 0.05
        jitter = np.clip(sigma * np.random.randn(len(vs), 3), -1 * clip, clip)
        vs += jitter

        # mirror transformation (YoZ plane), after centralization
        if np.random.rand(1) > 0.5:
            vs[:, 0] = -vs[:, 0]
            offsets[:, 0] = -offsets[:, 0]
            ts[:] = ts[:, [0, 2, 1]]        # keep the direction of normals
            idx1 = np.logical_and(ls<9, ls>0)
            idx2 = ls>=9
            ls[idx1] += 8
            ls[idx2] -= 8

        ts = np.roll(ts, np.random.randint(0, 3, 1), axis=1)        # change the order of vertices

        # rotation along the up axis
        # if self.rotation:
        #     vs = self.rotate(vs)

        return vs, ts, offsets, ls

    def rotate(self, vs):
        # randomly rotate the point clouds along the z axis.
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        return np.matmul(vs, rotation_matrix)

    def read_data(self, fps):
        pass

    @staticmethod
    def get_train_loader(args, n_gpus):
        dataset = DentalSegDataset(args, args.train_folder, True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize*n_gpus, shuffle=True,
                                                   num_workers=args.train_workers)  # single processor
        return train_loader

    @staticmethod
    def get_val_loader(args, n_gpus):
        dataset = DentalSegDataset(args, args.test_folder, False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize*n_gpus, shuffle=False,
                                             num_workers=args.test_workers)  # single processor
        return loader

import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset


class ST_VertexData(Dataset):

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
        vs, ts, y = torch.load(fp)
        vs -= vs.mean(0)        # preprocess
        if self.augmentation:
            vs, ts, y = self.augment(vs, ts, y)
        # sample
        # ts, y = sample(ts, y, self.num_points)
        idx = np.random.permutation(len(ts))[:self.num_points]
        if len(idx) < self.num_points:
            idx = np.concatenate((idx, idx), 0)[:self.num_points]
        ts = ts[idx]
        y = y[idx]

        vs = torch.tensor(vs, dtype=torch.float32)
        ts = torch.tensor(ts, dtype=torch.long)
        X = calc_features(vs, ts)
        y = torch.tensor(y, dtype=torch.long)

        return X, y, fp

    def augment(self, vs, ts, ls):
        # jitter vertices
        if np.random.rand(1) > 0.5:
            sigma, clip = 0.01, 0.05
            jitter = np.clip(sigma * np.random.randn(len(vs), 3), -1 * clip, clip)
            vs += jitter

        # mirror transformation (YoZ plane), after centralization
        if np.random.rand(1) > 0.5:
            vs[:, 0] = -vs[:, 0]
            ts[:] = ts[:, [0, 2, 1]]        # keep the direction of normals

        ts = np.roll(ts, np.random.randint(0, 3, 1), axis=1)        # change the order of vertices

        # rotation along the up axis
        # if self.rotation:
        #     vs = self.rotate(vs)

        return vs, ts, ls

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

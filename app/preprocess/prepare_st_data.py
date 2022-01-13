import os, glob, click
import torch
import numpy as np
import trimesh
from functools import partial
from multiprocessing import Pool
from mesh import TriMesh


def load_mesh(fp) -> trimesh.Trimesh:
    return trimesh.load(fp)

def save_st(fp, dst_path:str):
    case_id = os.path.split(os.path.dirname(fp))[-1]
    m = load_mesh(fp)
    ls = np.loadtxt(fp[:-3]+'txt', int)
    for l in np.unique(ls):
        if l > 0 and l % 10 != 8:
            out_fp = os.path.join(dst_path, case_id+'_%d.pth'%(l))
            if os.path.exists(out_fp):
                continue

            pts = m.triangles_center[ls==l]
            min_bbox, max_bbox = pts.min(0), pts.max(0)
            min_bbox = min_bbox - np.array([1., 1., 4.])
            max_bbox = max_bbox + np.array([1, 1, 4])

            idx = np.logical_and(np.all(m.triangles_center > min_bbox, 1), np.all(m.triangles_center<max_bbox, 1))
            tooth = trimesh.Trimesh(m.vertices, m.faces[idx])
            out_vs = np.array(tooth.vertices, dtype='f4')
            out_ts = np.array(tooth.faces)
            out_ls = ls[idx]
            out_ls[out_ls!=l] = 0
            out_ls[out_ls==l] = 1
            torch.save((out_vs, out_ts, out_ls), out_fp)


@click.command()
@click.option('--src_path', default='/mnt/shenkaidi/MyProjects/DentalModelProcessing/ToothSeg/v1/Experiments/data/data2/1_origin_data')
@click.option('--dst_path', default='/mnt/shenkaidi/MyProjects/DentalModelProcessing/SingleToothSeg/datasets/st_data2/train')
def run(src_path, dst_path):
    fps = glob.glob(src_path+'/*/*.stl')
    func = partial(save_st, dst_path=dst_path)

    with Pool(6) as p:
        p.map(func, fps)

if __name__ == '__main__':
    run()

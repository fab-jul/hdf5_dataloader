import glob
import math
import pickle

import dataset

import pytest
import os
import maker

import glob
from dataset import HDF5Dataset
from transforms import ArrayToTensor, ArrayCenterCrop
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


@pytest.fixture()
def out_dir_with_shards(tmpdir):
    out_dir = str(tmpdir.mkdir('out'))
    num_per_shard = 7
    inp_glob = 'inp/*.jpg'
    num_inps = len(glob.glob(inp_glob))
    assert num_inps > 0  # Assume some folder with images at inp
    assert num_inps % num_per_shard != 0
    with pytest.raises(ValueError):
        maker.main([out_dir, inp_glob, '--shuffle', '--num_per_shard', str(num_per_shard)])
    maker.main([out_dir, inp_glob, '--shuffle', '--num_per_shard', str(num_per_shard), '--force'])
    num_shards = len(glob.glob(os.path.join(out_dir, 'shard_*.hdf5')))
    assert num_shards == math.ceil(num_inps / num_per_shard)
    return out_dir

def test_maker(out_dir_with_shards):
    assert os.path.isdir(out_dir_with_shards)
    assert os.path.isfile(os.path.join(out_dir_with_shards, 'log'))
    assert os.path.isfile(os.path.join(out_dir_with_shards, maker.NUM_PER_SHARD_PKL))


def test_dataset(out_dir_with_shards):
    transform_hdf5 = transforms.Compose([ArrayCenterCrop(64), ArrayToTensor()])
    # Check nice exceptions for invalid inputs
    with pytest.raises(ValueError):
        HDF5Dataset([], transform=transform_hdf5)
    with pytest.raises(ValueError):
        HDF5Dataset(['doesnotexists.hdf5'], transform=transform_hdf5)
    all_file_ps = sorted(glob.glob(os.path.join(out_dir_with_shards, '*.hdf5')))
    ds = HDF5Dataset(all_file_ps, transform=transform_hdf5)

    # using the standard PyTorch DataLoader
    dl = DataLoader(ds, batch_size=3, shuffle=True, num_workers=0)

    # last should be smaller
    assert dataset.get_num_in_shard(all_file_ps[0]) > dataset.get_num_in_shard(all_file_ps[-1])
    # sum number of imgs in all shards all except last shard
    expected_num_imgs = sum(dataset.get_num_in_shard(shard_p) for shard_p in all_file_ps[:-1])
    actual_num_imgs = 0
    for batch in dl:
        actual_num_imgs += batch.shape[0]
    assert actual_num_imgs == expected_num_imgs


def test_dataset_without_pickle(out_dir_with_shards):
    os.remove(os.path.join(out_dir_with_shards, maker.NUM_PER_SHARD_PKL))
    transform_hdf5 = transforms.Compose([ArrayCenterCrop(64), ArrayToTensor()])
    all_file_ps = sorted(glob.glob(os.path.join(out_dir_with_shards, '*.hdf5')))
    HDF5Dataset(all_file_ps, transform=transform_hdf5)


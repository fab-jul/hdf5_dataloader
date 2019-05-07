import os
import random
import pickle
import h5py
from torch.utils.data import Dataset
from maker import NUM_PER_SHARD_PKL


default_opener = lambda p_: h5py.File(p_, 'r')


class HDF5Dataset(Dataset):
    def __init__(self, file_ps,
                 transform,
                 shuffle_shards=True,
                 opener=default_opener,
                 seed=123):
        """
        :param file_ps: list of file paths to .hdf5 files. Last (alphabetically) file is expected to contain less
        images.
        :param transform: transformation to apply to read HDF5 dataset. Must contain some transformation to array!
        See README.md
        :param shuffle_shards: if true, shards are shuffled with seed
        """
        if transform is None:
            raise ValueError('transform must have at least hdf5.transforms.HDF5DatasetToArray()')
        if len(file_ps) == 0 or not all(os.path.isfile(p) for p in file_ps):
            raise ValueError('Expected list of paths to HDF5 files, got {}'.format(file_ps))
        self.opener = opener
        self.ps, self.num_per_shard = HDF5Dataset.filter_smaller_shards(file_ps)
        if shuffle_shards:
            r = random.Random(seed)
            r.shuffle(self.ps)
        self.transform = transform

    def __len__(self):
        return len(self.ps) * self.num_per_shard

    def __getitem__(self, index):
        shard_idx = index // self.num_per_shard
        idx_in_shard = index % self.num_per_shard
        shard_p = self.ps[shard_idx]
        with self.opener(shard_p) as f:
            el = f[str(idx_in_shard)]
            el = self.transform(el)  # must turn to array
        return el

    @staticmethod
    def filter_smaller_shards(file_ps, opener=default_opener):
        """
        Filter away the (alphabetically) last shard, which is assumed to be smaller. This function also double checks
        that all other shards have the same number of entries.
        :param file_ps: list of .hdf5 files, does not have to be sorted.
        :param opener: function to open shards
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        assert file_ps, 'No files given'
        file_ps = sorted(file_ps)  # we assume that smallest shard is at the end
        num_per_shard_prev = None
        ps = []
        for i, p in enumerate(file_ps):
            num_per_shard = get_num_in_shard(p, opener)
            if num_per_shard_prev is None:  # first file
                num_per_shard_prev = num_per_shard
                ps.append(p)
                continue
            if num_per_shard_prev < num_per_shard:
                raise ValueError('Expected all shards to have the same number of elements,'
                                 'except last one. Previous had {} elements, current ({}) has {}!'.format(
                                    num_per_shard_prev, p, num_per_shard))
            if num_per_shard_prev > num_per_shard:  # assuming this is the last
                is_last = i == len(file_ps) - 1
                if not is_last:
                    raise ValueError(
                            'Found shard with too few elements, and it is not the last one! {}\n'
                            'Last: {}\n'.format(p, file_ps[-1]))
                print('Filtering shard {}, dropping {} elements...'.format(p, num_per_shard))
                break  # is last anyways
            else:  # same numer as before, all good
                ps.append(p)
        return ps, num_per_shard_prev


def get_num_in_shard(shard_p, opener=default_opener):
    hdf5_root = os.path.dirname(shard_p)
    p_to_num_per_shard_p = os.path.join(hdf5_root, NUM_PER_SHARD_PKL)
    # Speeds up filtering massively on slow file systems...
    if os.path.isfile(p_to_num_per_shard_p):
        with open(p_to_num_per_shard_p, 'rb') as f:
            p_to_num_per_shard = pickle.load(f)
            num_per_shard = p_to_num_per_shard[os.path.basename(shard_p)]
    else:
        print('\rOpening {}...'.format(shard_p), end='')
        with opener(shard_p) as f:
            num_per_shard = len(f.keys())
    return num_per_shard

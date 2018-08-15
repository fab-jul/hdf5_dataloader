import h5py
import argparse
import random
import glob
import numpy as np
import os
import itertools as it
from collections import defaultdict
from fjcommon import functools_ext as ft
from fjcommon import timer

from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as multiprocessing


_ASSERT_ASSUMPTIONS = True
_LOG_ACCESS = False


default_opener = lambda p_: h5py.File(p_, 'r')


# TODO:
# - document excessive memory usage
# - verbose flag for shard reader
# - chunked shuffle loading
# - document expected shard format: keys '0', ..., 'num_per_shard'


class HDF5DataLoader(DataLoader):
    """
    Main Class. Use it to create a DataLoader directly.
    To start a new epoch, use the `reshuffle` method.
    """
    def __init__(self, hdf5_glob, transforms, batch_size, num_workers):
        # TODO: shuffle shards on new iter!
        self.dataset = _HDF5Dataset(hdf5_glob, transforms, batch_size, num_readers=num_workers)
        super(HDF5DataLoader, self).__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers)

    def reshuffle(self):
        self.dataset.shard_reader.initialize(shuffle_shards=True)


# ------------------------------------------------------------------------------


class _HDF5Dataset(Dataset):
    def __init__(self, hdf5_glob=None, transform=None, batch_size=None, num_readers=1, shuffle_shards=True):
        self.transform = transform
        with timer.execute('glob(...)'):
            all_file_ps = glob.glob(hdf5_glob)
        self.shard_reader = ShardReader(all_file_ps, num_readers, batch_size, filter_smaller_shards=True,
                                        shuffle_shards=shuffle_shards)
        self.num_els = self.shard_reader.max_full_batches * batch_size

    def __len__(self):
        return self.num_els

    def __getitem__(self, idx):
        el = self.shard_reader[idx]
        if self.transform:
            el = self.transform(el)
        return el


class ShardReader(object):
    @staticmethod
    def filter_smaller_shards(file_ps, opener=default_opener):
        """
        TODO: note that this assumes file_ps is sorted
        """
        num_per_shard_prev = None
        for i, p in enumerate(file_ps):
            with opener(p) as f:
                num_per_shard = len(f.keys())
            if num_per_shard_prev is None:
                num_per_shard_prev = num_per_shard
                yield p
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
                            'Last: {}\n'
                            'Make sure to sorte file_ps before filtering.'.format(p, file_ps[-1]))
                print('Filtering shard {}, dropping {} elements...'.format(p, num_per_shard))
                break  # is last anyways
            else:  # same numer as before, all good
                yield p

    def __init__(self, file_ps, num_readers, batch_size,
                 filter_smaller_shards=False, assert_exactly_once=False,
                 shuffle_shards=True,
                 opener=default_opener):
        if assert_exactly_once:
            assert not filter_smaller_shards, 'filter_smaller_shards should be False if assert_exactly_once=True'
        if filter_smaller_shards:
            # The filter assumes that paths are sorted
            file_ps = ShardReader.filter_smaller_shards(sorted(file_ps), opener)

        with timer.execute('filter'):
            self.file_ps = list(file_ps)
        self.files = None  # reader -> [shard_f]
        self.num_readers = num_readers
        self.batch_size = batch_size
        self.opener = opener
        self.assert_exactly_once = assert_exactly_once

        self.num_per_shard = None
        self.max_full_batches, self.total_els_dropped = None, None

        self.initialize(shuffle_shards)

    def initialize(self, shuffle_shards):
        """"""
        if self.files is not None:
            for shard_f in ft.concat(self.files.values()):
                shard_f.close()  # might close same file twice if it was duplicated, but this is not a problem
            self.files = None

        if shuffle_shards:
            print('Shuffling shards...')
            random.shuffle(self.file_ps)
        else:
            self.file_ps = sorted(self.file_ps)

        self.files = defaultdict(list)  # reader -> [shard_f]
        for r, shard_p in zip(it.cycle(range(self.num_readers)), self.file_ps):
            self.files[r].append(self.opener(shard_p))
        assert any(len(files_r) > 0 for files_r in self.files.values()), 'No files!'

        # some readers may have 1 less file then others (never more)
        # for those, pick a random file and append it as a final file to read from
        # this ensures that all readers can read from the same number of files and thus from the same number of
        # examples.
        # However, this yields to up to (K-1) * S examples being read twice, where K=number_readers, S=num_per_shard.
        max_file_count = max(len(files_r) for files_r in self.files.values())
        for r, files_r in self.files.items():
            if len(files_r) == max_file_count:
                continue
            if len(files_r) == max_file_count - 1:
                assert not self.assert_exactly_once, \
                    ('Duplicating some files because not every reader received equal number of elements. '
                     'Try setting num_readers = 1 or make sure that the number of files is divisible by the '
                     'number of readers.')
                random_file_of_r = random.choice(files_r)
                print('Duplicating {} els'.format(len(random_file_of_r.keys())))
                files_r.append(random_file_of_r)
            else:
                raise ValueError('{}, {} {}'.format(r, len(files_r), max_file_count))

        if _LOG_ACCESS:
            for r, files_r in self.files.items():
                num_els = sum(len(shard_f.keys()) for shard_f in files_r)
                print('READER {}: {} files // {} els // {} batches'.format(
                        r, len(files_r), num_els, num_els / self.batch_size))

        self.num_per_shard = self._get_num_per_shard(self.files)
        self.max_full_batches, self.total_els_dropped = self._get_max_full_batches(
                self.files, self.num_per_shard, self.batch_size)
        assert not self.assert_exactly_once or self.total_els_dropped == 0, \
            ('Dropping {} elements not fitting into a batch of a reader. Make sure that for all readers r, '
             'the sum of elements in the files of the reader r is divisible by the batch_size.'.format(
                    self.total_els_dropped))

    @staticmethod
    def _get_num_per_shard(files):
        all_shard_fs = files.values()  # [[shard_f]]
        all_shard_fs = ft.concat(all_shard_fs)  # [shard_f]
        nums_per_shard = set(len(shard_f.keys()) for shard_f in all_shard_fs)
        if len(nums_per_shard) != 1:
            raise ValueError('Expected all shards to have same number of elements. Use '
                             'ShardReader.filter_smaller_shards.')
        return nums_per_shard.pop()

    @staticmethod
    def _get_max_full_batches(files, num_per_shard, batch_size):
        total_els_dropped = 0
        max_full_batches = 0
        for r, shard_fs_of_reader in files.items():
            num_els_reader = len(shard_fs_of_reader) * num_per_shard
            els_not_fitting_in_batch = num_els_reader % batch_size
            total_els_dropped += els_not_fitting_in_batch
            max_idx_of_reader = num_els_reader - els_not_fitting_in_batch
            max_full_batches += max_idx_of_reader // batch_size
        return max_full_batches, total_els_dropped


    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        reader_idx = batch_idx % self.num_readers
        idx_within_reader = (
            (idx // (self.batch_size * self.num_readers)) * self.batch_size +
            idx % self.batch_size
        )
        shard_idx_within_reader = idx_within_reader // self.num_per_shard
        idx_in_shard = idx_within_reader % self.num_per_shard
        assert shard_idx_within_reader < len(self.files[reader_idx]), (
            '{: 3d} -> READER {: 3d} // IDX_R {: 3d} // SHARD_IDX_R {: 3d} -> ...[{: 3d}]'.format(
                    idx, reader_idx, idx_within_reader, shard_idx_within_reader, idx_in_shard),
            self.files[reader_idx]
        )
        shard_f = self.files[reader_idx][shard_idx_within_reader]
        el = shard_f[str(idx_in_shard)]  # TODO: assumes shards are indexed with strings, from 0...num_per_shard-1
        if _LOG_ACCESS:
            print('{: 3d} -> READER {: 3d} // IDX_R {: 3d} // SHARD_IDX_R {: 3d} -> /...[{: 3d}] = {}'.format(
                    idx, reader_idx, idx_within_reader, shard_idx_within_reader, idx_in_shard, el))
        return el



def test():
    shuffle_dataset = True
    p = 'test.hdf5'
    if os.path.isfile(p):
        os.remove(p)

    with h5py.File(p, "w") as f:
        root = f.create_group('root')
        num_files = 10
        idx_mapping = np.arange(num_files)
        if shuffle_dataset:
            np.random.shuffle(idx_mapping)
        for i, file_i in enumerate(idx_mapping):
            h, w = np.random.randint(256, 512, 2)
            m = np.random.randint(0, 255, 3 * h * w).reshape((3, h, w)).astype(np.uint8)
            m[0, 0, 0] = file_i  # ...
            root.create_dataset(str(i), data=m, shuffle=True)

    with h5py.File(p, 'r') as f:
        print(list(f.keys()))
        root = f['root']
        print(root['1'])
        print(root['1'][0, 0, 0])

    dl = Foo(None, None, 4, 4)
    for e, ids in dl:
        print(e, ids)


# ------------------------------------------------------------------------------

import operator


#TODO: Move to fjcommon
class _Acc(object):
    def __init__(self, iterator, reduce=operator.add, total_start=0):
        self.iterator = iterator
        self.reduce = reduce
        self.total = total_start

    def __iter__(self):
        for el in self.iterator:
            self.total = self.reduce(self.total, el)
            yield el


def _show_stats(hdf5_glob):
    ps = sorted(glob.glob(hdf5_glob))
    assert len(ps) > 0, 'No matches for {}'.format(hdf5_glob)
    ns = _Acc(_get_num_entries(ps))
    for p, n in zip(ps, ns):
        print('{}: {} elements'.format(p, n))
    print('-- Total: {}'.format(ns.total))


def _get_num_entries(ps):
    for p in ps:
        with h5py.File(p, 'r') as f:
            print(sorted(f.keys()))
            yield len(f.keys())



def _rename(hdf5_glob):
    print('Renaming...')
    ps = sorted(glob.glob(hdf5_glob))
    assert len(ps) > 0, 'No matches for {}'.format(hdf5_glob)
    num_per_shard = None
    for p in ps:
        with h5py.File(p, 'r+') as f:
            num_per_shard_cur = len(f.keys())
            if num_per_shard is None:
                num_per_shard = num_per_shard_cur
                continue
            orig_keys = set(f.keys())
            for k in orig_keys:
                k_int = int(k)
                new_k_int = k_int % num_per_shard
                assert str(new_k_int) not in orig_keys
                print('{} -> {}'.format(k_int, new_k_int))
                f[str(new_k_int)] = f[k]
                del f[k]


# ------------------------------------------------------------------------------


class DummyDataset(object):
    def __init__(self, num_workers, batch_size):
        self.num_workers = num_workers
        self.batch_size = batch_size

        self._log_empty_value = -1
        self._log = multiprocessing.Array('i', [self._log_empty_value] * num_workers)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        reader_idx = (idx // self.batch_size) % self.num_workers
        worker_id = os.getpid()

        if self._log is not None:
            if self._log[reader_idx] == self._log_empty_value:
                self._log[reader_idx] = worker_id
            else:
                assert self._log[reader_idx] == worker_id

        return reader_idx, worker_id, idx  # return for _testt


class _Checka(object):
    """
    Checks the following assumptions:
    - a whole batch is read by the same reader
    - every worker has its own PID
    - the workers take turns loading data
    """
    def __init__(self):
        self.pids_to_idxs = defaultdict(set)

    def check(self, i, el, check_every=100):
        idxs, pids = el[0], el[1]
        pids = set(pids.detach().cpu().numpy())
        assert len(pids) == 1
        pid = pids.pop()
        idxs = set(idxs.detach().cpu().numpy())
        self.pids_to_idxs[pid] |= idxs
        if i % check_every == 0:
            assert all(len(idxs) <= 1 for idxs in self.pids_to_idxs.values())
            assert len(self.pids_to_idxs.keys()) == len(set(self.pids_to_idxs.keys()))
            assert len(self.pids_to_idxs.values()) == len(set((next(iter(idxs))
                                                               for idxs in self.pids_to_idxs.values()))), \
                self.pids_to_idxs
        return el[2:]



def _test_checka():
    """
    assures that
        reader_idx = (idx // self.batch_size) % self.num_workers
        worker_id = os.getpid() % self.num_workers
    are in one to one correspondence
    """
    import sys
    num_workers = int(sys.argv[1])
    batch_size = 30
    dd = DummyDataset(num_workers, batch_size)
    dl = DataLoader(dd, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    checka = _Checka()
    for i, el in enumerate(dl):
        el = checka.check(i, el)
        print(el)


# ------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument('hd5f_glob')
    flags = p.parse_args()
    _show_stats(flags.hd5f_glob)


if __name__ == '__main__':
    main()



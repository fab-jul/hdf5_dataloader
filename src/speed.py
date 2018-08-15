import itertools as it
import resource
import random
import torch.nn as nn
import os
from hdf5.dataloader import HDF5DataLoader
import torchvision.transforms as transforms
from dataloaders.images_loader import ImagesDataset
from torch.utils.data import DataLoader
import socket
import logdir_helpers
from tensorboardX import SummaryWriter
from fjcommon import timer
import pytorch_ext as pe


# TODO:
# put in numbers from speedtest on bmicgpu05
from hdf5.transforms import ArrayToTensor, ArrayCenterCrop


feed_through_network = [False, True]
batch_size = 5
transform_hdf5 = transforms.Compose([ArrayCenterCrop(256), ArrayToTensor()])
transform_pil = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
taskset = os.environ.get('TASKSET', '1') or '1'
num_workers = max(map(int, filter(None, taskset.split('-')))) * 2
print('taskset={} -> {} workers'.format(taskset, num_workers))


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Conv2d(3, 10, 3)
        self.pool = nn.AvgPool2d(16, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


def speed_test():
    speed_test_hdf5()


def print_memory_usage():
    print('Usage: {}, {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))


def speed_test_hdf5():
    print('speed_test_hdf5')
    hdf5_globs = [
        '/srv/glusterfs/mentzerf/coco/hd5f/train2017/*.hdf5',
        '/scratch/mentzerf/coco/hd5f/train2017/*.hdf5',
#        '/srv/glusterfs/mentzerf/coco/train2017',
#        '/scratch_net/bmicgpu05/mentzerf/coco/train2017/'
    ]

    tests = list(it.product(feed_through_network, hdf5_globs))
    for trial in range(4):
        random.shuffle(tests)
        for feed_through_network_cur, p in tests:
            is_glusterfs = 'glusterfs' in p
            is_hdf5 = 'hdf5' in p
            logdir = get_log_dir(is_hdf5, feed_through_network_cur, is_glusterfs,
                                 temporary='TMPJOB' in os.environ)

            t = timer.TimeAccumulator()
            sw = SummaryWriter(logdir)
            print(os.path.basename(logdir))

            if is_hdf5:
                print('Preparing HDF5 dataloader...')
                dl = HDF5DataLoader(p, transform_hdf5, batch_size, num_workers)
            else:
                print('Preparing Images dataloader...')
                ds = ImagesDataset(
                        p,
                        transform=transform_pil,
                        cache_p='/home/mentzerf/phd_code/pytorch_tut/image_cache.pkl',
                        min_size=256)
                dl = DataLoader(ds, batch_size, shuffle=True, num_workers=num_workers)

            if feed_through_network_cur:
                print('Creating Network...')
                net = Network().to(pe.DEVICE)
            else:
                net = None

            print_memory_usage()
            print('Starting...')
            dl_it = iter(dl)
            print_memory_usage()
            max_iter = 2500 * num_workers
            print('Checking <={} batches'.format(max_iter))

            for i in range(len(dl)):
                if i == max_iter:
                    break
                with t.execute():
                    batch = next(dl_it)
                    if net is not None:
                        batch = batch.to(pe.DEVICE)
                        batch = batch.float()
                        batch = net(batch)
                if i % 30 == 0:
                    print(batch.shape, batch.dtype)
                    mean = batch_size/t.mean_time_spent()
                    print('{: 10d} {:.3f} {} (Trial {})'.format(i, mean, os.path.basename(logdir), trial))

                    sw.add_scalar('Mean', mean, i)
                    sw.file_writer.flush()
            sw.close()
            print('Test completed')
        print('All completed')


def get_log_dir(hdf5, feed_through_network_cur, gluster_fs, temporary=False):
    description = ['{}={}'.format(name, value) for name, value in [
        ('H', int(hdf5)),
        ('gl', int(gluster_fs)),
        ('W', int(num_workers)),
        ('net', int(feed_through_network_cur)),
    ]]
    return logdir_helpers.create_unique_log_dir([socket.gethostname()] + description,
                                                'speed_test_logs' + ('_TMP' if temporary else ''))


def main():
    speed_test()


if __name__ == '__main__':
    main()

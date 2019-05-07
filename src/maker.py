import argparse
import sys
import glob
import pickle
import time
import shutil
import numpy as np
import random
import h5py
import os
import printing
from PIL import Image


NUM_PER_SHARD_PKL = 'num_per_shard.pkl'


def make_hdf5_files(out_dir, images_glob, shuffle=True, num_per_shard=1000, max_shards=None,
                    min_size=None, name_fmt='shard_{:010d}.hdf5', force=False):
    """
    Notes:
        - total output file size may be much bigger, as JPGs get decompressed and stored as uint8
    Parameters mirror arguments of maker.py, see main() below.
    """
    if os.path.isdir(out_dir):
        if not force:
            raise ValueError('{} already exists.'.format(out_dir))
        print('Removing {}...'.format(out_dir))
        time.sleep(1)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'log'), 'w') as f:
        info_str = '\n'.join('{}={}'.format(k, v) for k, v in [
            ('out_dir', out_dir),
            ('images_glob', images_glob),
            ('shuffle', shuffle),
            ('num_per_shard', num_per_shard),
            ('max_shards', max_shards),
            ('min_size', min_size),
            ('name_fmt', name_fmt),
            ('force', force)])
        print(info_str)
        f.write(info_str + '\n')

    print('Getting images...')
    image_ps = sorted(glob.glob(images_glob))
    assert len(image_ps) > 0, 'No matches for {}'.format(images_glob)
    print('Found {} images'.format(len(image_ps)))

    if shuffle:
        print('Shuffling...')
        random.shuffle(image_ps)
    num_shards_total = str(len(image_ps)//num_per_shard)

    if min_size:
        print('Filtering for >= {}...'.format(min_size))
        image_ps = (p for p in image_ps if _big_enough(p, min_size))
        num_shards_total = '<=' + num_shards_total

    writer = None
    count = 0
    shard_ps = []
    with printing.ProgressPrinter() as progress_printer:
        for count, image_p in enumerate(image_ps):
            if count % num_per_shard == 0:
                progress_printer.finish_line()
                if writer:
                    writer.close()
                shard_number = count // num_per_shard
                if max_shards is not None and shard_number == max_shards:
                    print('Created {} shards...'.format(max_shards))
                    return
                shard_p = os.path.join(out_dir, name_fmt.format(shard_number))
                assert not os.path.exists(shard_p), 'Record already exists! {}'.format(shard_p)
                print('Creating {} [{}/{}]...'.format(shard_p, shard_number, num_shards_total))
                writer = h5py.File(shard_p, 'w')
                shard_ps.append(shard_p)
            image = Image.open(image_p).convert('RGB')
            image = np.array(image, np.uint8).transpose((2, 0, 1))
            assert image.shape[0] == 3
            index = str(count % num_per_shard)  # expected by HDF5DataLoader, TODO: document
            writer.create_dataset(index, data=image)
            progress_printer.update((count % num_per_shard) / num_per_shard)
    if writer:
        writer.close()
        assert len(shard_ps)
        # writing this to num_per_shard.pkl
        p_to_num_per_shard = {os.path.basename(shard_p): num_per_shard for shard_p in shard_ps}
        last_shard_p = shard_ps[-1]
        with h5py.File(last_shard_p, 'r') as f:
            p_to_num_per_shard[os.path.basename(last_shard_p)] = len(f.keys())
        with open(os.path.join(out_dir, NUM_PER_SHARD_PKL), 'wb') as f:
            pickle.dump(p_to_num_per_shard, f)
    else:
        print('Nothing written, processed {} files...'.format(count))


def _big_enough(image_p, min_size):
    img = Image.open(image_p)
    img_min_dim = min(img.size)
    if img_min_dim < min_size:
        print('Skipping {} ({})...'.format(image_p, img.size))
        return False
    return True


def main(args):
    p = argparse.ArgumentParser()
    p.add_argument('out_dir',
                   help='Where to store .hdf5 files. Additionally, the following files are stored: log, which saves '
                        'the parameters used to create the .hdf5 files, and num_per_shard.pkl, which stores a '
                        'dictionary mapping file names to number of entries in that file (see '
                        'get_num_in_shard in dataset.py).')
    p.add_argument('images_glob',
                   help='Glob to images to use. Make sure to escape, e.g., '
                        'path/to/imgs/\\*.png or "path/to/imgs/*.png"')
    p.add_argument('--shuffle', action='store_true',
                   help='Shuffle images before putting them into records. Default: Not set')
    p.add_argument('--num_per_shard', type=int, default=1000,
                   help='Number of images per record. Default: 1000')
    p.add_argument('--max_shards', type=int,
                   help='Maximum number of shards. Default: None')
    p.add_argument('--min_size', type=int,
                   help='Only use images with either height or width >= MIN_SIZE. Default: None')
    p.add_argument('--name_fmt', default='shard_{:010d}.hdf5',
                   help='Format string for shards, must contain one placeholder for number. '
                        'Default: shard_{:010d}.hdf5')
    p.add_argument('--force', action='store_true',
                   help='If given, continue creation even if `out_dir` exists already. NOTE: In this case, '
                        '`out_dir` is removed first!')
    flags = p.parse_args(args)
    make_hdf5_files(**flags.__dict__)


if __name__ == '__main__':
    main(sys.argv[1:])



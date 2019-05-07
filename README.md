# hdf5_dataloader
DataLoader subclass for PyTorch to work with HDF5 files. 

# Requirements

- h5py
- Pillow
- PyTorch (Tested with 0.4 and 1.0)
- Python 3 (Tested with 3.6)

# Usage

## Create .hdf5 files

    
```bash    
cd src
    
# Note the escaped *, as it is parsed in Python
python maker.py out_dir some/path/\*.png --shuffle --num_per_shard 500
```
    
More options are available, see `python maker.py --help`.

## Using DataLoader

```python
import glob
from hdf5_dataloader.dataset import HDF5Dataset
from hdf5_dataloader.transforms import ArrayToTensor, ArrayCenterCrop
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# create transform
# Note: cannot use default PyTorch ops, because they expect PIL Images
transform_hdf5 = transforms.Compose([ArrayCenterCrop(256), ArrayToTensor()])

# create dataset
all_file_ps = glob.glob('dataset/train/*.hdf5')
ds = HDF5Dataset(all_file_ps, transform=transform_hdf5)

# using the standard PyTorch DataLoader
dl = DataLoader(ds, batch_size=30, shuffle=True, num_workers=8)

# use
for batch in dl:
    ...
````

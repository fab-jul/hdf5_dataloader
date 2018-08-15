# hdf5_dataloader
DataLoader subclass for PyTorch to work with HDF5 files (BETA)

*This is work in progress. Documentation coming up.*

# Requirements

- h5py
- fjcommon (`pip install fjcommon`)
- TODO

# Usage

## Create .hdf5 files

    
```bash    
    cd src
    
    # Note the escaped *, as it is parsed in Python
    python maker.py out_dir some/path/\*.png --shuffle --num_per_shard 500
    
    # See python maker.py --help
```

## Using DataLoader

```python
    from hdf5_dataloader.dataloader import HDF5DataLoader
    from hdf5_dataloader.transforms import ArrayToTensor, ArrayCenterCrop
    import torchvision.transforms as transforms
    
    # create transform
    # Note: cannot use default PyTorch ops, because they expect PIL Images
    transform_hdf5 = transforms.Compose([ArrayCenterCrop(256), ArrayToTensor()])
    
    # create dataset
    dl = HDF5DataLoader(p, transform_hdf5, batch_size=30, num_workers=4)
    
    # use
    for batch in dl:
        ...
````

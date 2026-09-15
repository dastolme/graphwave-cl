import data_loader as dl

data_man = dl.HDF5GraphWaveDataset("./data/dataset.hdf5")
data_man._compute_node_stats()
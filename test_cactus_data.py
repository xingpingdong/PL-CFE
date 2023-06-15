import numpy as np
import h5py
split='train'
for split in ['train','test', 'val']:
    cactus_data = np.load('/media/dxp/Expansion/backup-20230121/fsl_data/cactus/acai_encodings/omniglot_256_{}.npz'.format(split))
    X = cactus_data['X']
    Y = cactus_data["Y"]
    np.savez('data/omniglot/omniglot_cache_{}.npz'.format(split), X=X, Y=Y)
# cfe_data = np.load('./data/cfe_encodings/omniglot_64_K_500_{}.npz'.format(split))
# X_cfe = cfe_data['X']
#
# data = h5py.File('data/omniglot/data.hdf5', 'r')

print(("  "))
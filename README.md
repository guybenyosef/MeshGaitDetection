# asu_gait


### Mesh model based on I2LMeshNet:
Clone I2LMeshNet from: https://github.com/mks0601/I2L-MeshNet_RELEASE, 
and place it in the `I2LMeshNet` folder. Add `__init__.py` to all folders.

Troubleshoot -- after installing `torchgeometry` (e.g., using conda) you may need to do the following: 
in `/home/guy/anaconda3/envs/py39/lib/python3.9/site-packages/torchgeometry/core/conversions.py`
Notice the following change:
```python
    mask_c1 = mask_d2 * ~mask_d0_d1
    # Guy -- used to be: mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = ~mask_d2 * mask_d0_nd1
    # Guy -- used to be: mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    # Guy -- used to be: mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
```


### Detect mesh and pose:
Run mesh on image:
```bash
python run_mesh.py -v Samples/image2.jpg
```

Run mesh on all jpg images in folder: 
```bash
python run_mesh.py -v /path/to/folder/
```

### Gait embedding:
Get embedding from pose-per-frame. Requires a path to folder with kpts.npz files, computed by run_mesh. 
```bash
python get_embedding.py -v /path/to/folder/with/kpts.npz/files
```

### Datasets:
CASIA:
http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp

OL-LP-Bag:
http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPBag.html

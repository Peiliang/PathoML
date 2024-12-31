## Install dependencies

The below command creates a conda environment named unified_env in your working directory.

```bash
conda env create -f unified_env.yml
pip install mpi4py
pip install pydicom
pip install nibabel
conda activate unified_env
```

## Model Checkpoints for CellVit

Model checkpoints can be downloaded here:

- [CellViT-SAM-H](https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV) 

Please put this checkpoint under "/home/peiliang/projects/PathoML/PatchToPathoML/Cellvit/checkpoint"
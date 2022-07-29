# Graph neural networks for PDAC vs CP in histology


## Conda environment
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
Required packages
```
  $ conda env create --name pyg --file env.yml
  $ conda activate pyg
```

## Prepare patches
Download the dataset from [Google Drive](https://drive.google.com/file/d/1ZRNYaju0GvT22pw7BXrWI5t1q4DdKgft/view?usp=sharing)  
Use the `deepzoom_tiler.py` from [this repository](https://github.com/binli123/dsmil-wsi) to prepare patches. 
Put the patches into a directory named `data_bags/TMA`.  
Otherwise, the patches (normalzed) can be downloaded from [Google Drive](https://drive.google.com/file/d/1dITGpox7RsXVNaMAsMHF0N27MjNKcC_3/view?usp=sharing). 

## Stain normalization in batches 
Example usage is shown in `normalization.ipynb`.  
Note that the normalization algorithm is from [Paper](https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf)
and the code is from [GitHub](https://github.com/schaugf/HEnorm_python).

## Computing features using pretrained ResNet
See [notebook_distributed.ipynb](https://github.com/uw-loci/gnn-pccp/blob/master/notebook_distributed.ipynb).

## Training and cross-validation
See [notebook_distributed.ipynb](https://github.com/uw-loci/gnn-pccp/blob/master/notebook_distributed.ipynb).

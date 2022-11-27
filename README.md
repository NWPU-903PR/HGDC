# A novel heterophilic graph diffusion convolutional network for identifying cancer driver genes
*by Tong Zhang, Shao-Wu Zhang\* (zhangsw@nwpu.edu.cn), Mingyu Xie *

## Requirements

```shell
numpy
pandas
scikit-learn
python==3.7.11
torch==1.8.0
torch-geometric==2.0.2
```

## Usage
Run main.py script to conduct ten times 5CV on the network of GGNet, PathNet, or PPNet.
Run train_model.py script to obtain predicted scores of genes in the network of GGNet, PathNet, or PPNet, and the predicted scores of genes are saved in "./results/"

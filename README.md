# Spatiotemporal Hypergraph Convolution Network for Stock Movement Forecasting

This codebase contains the python scripts for STHGCN, the model for the ICDM 2020 paper [link](https://ieeexplore.ieee.org/document/9338303).

## Environment & Installation Steps
Python 3.6, Pytorch, Pytorch-Geometric and networkx.

## Dataset and Preprocessing 

Download the dataset and follow preprocessing steps from [here](https://github.com/dmis-lab/hats). 
```bash
bash download.sh
```

## Run

Execute the following python command to train STHGCN: 
```bash
make test_phase=1 save_dir=save
```
test_phase : phase that you want to test

## Cite
Consider citing our work if you use our codebase

```c
@INPROCEEDINGS{9338303,  author={Sawhney, Ramit and Agarwal, Shivam and Wadhwa, Arnav and Shah, Rajiv Ratn},  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},   title={Spatiotemporal Hypergraph Convolution Network for Stock Movement Forecasting},   year={2020},  volume={},  number={},  pages={482-491},  doi={10.1109/ICDM50108.2020.00057}}
```


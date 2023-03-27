# TemT
A New Temporal Knowledge Graph Completion Method with Transformer Hawkes Process and Random Walk Aggregation Strategy.

### Requirements
+ Python3 (tested on 3.9)
+ PyTorch (tested on 1.13)
+ CUDA (tested on 11.6)
+ dql (tested on 0.9.1)

### How to use
After installing the requirements, run the following command to preprocess datasets.:
```commandline
python3 data/DATA_NAME/get_history.py
python3 data/DATA_NAME/get_history_tpre_appro.py
```
To train and test the model:
```commandline
python3 co-train.py -d DATA_NAME
```
Only evaluating the model:
```commandline
python3 python3 co-train.py -d DATA_NAME --only_eva true --eva_dir MODEL_DIR
```

### Citation
If you use the codes, please cite the following paper:

### License
This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

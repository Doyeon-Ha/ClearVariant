# ClearVariant

## How model will stored
The result of task will be stored under
`data/result/model/{task}/{db}/{db_processing}.{db_option}/{pretrained_model}/{date}`.

For instance, if you run main.py with default config, 
`data/result/model/train/itanlab/stratified_5fold.itanlab/esm2_t6_8M_UR50D/{date}` will be made.

The structure of directory is below.
Some files are only created for certain tasks.
```
data/input/
├── attn_out/ # default
│   └── test_{sample_num}.npz # task_write
├── model_param/ # default
│   └── {epoch}/ # task_train
│       ├── config.json
│       └── model.safetensors
├── config.yaml # default
├── classification_result.tsv # task_inference
├── dataset.csv # default
└── log.log # default
```
## Anaconda setup
Once Anaconda is installed, you can create a new environment with:
```shell
conda create --name ocr python=3.7
```

## Install locally
Once you have created your Python environment you can simply type:
```shell
git clone git@github.com:hyc2026/ocr.git
cd ocr
pip install -r requirements.txt
```

## Train
1. 修改./config中对应算法的yaml中参数，基本上只需修改数据路径即可。
2. 运行下面命令

```shell
python3 tools/det_train.py --config ./config/det_DB_mobilev3.yaml --log_str train_log  --n_epoch 1200 --start_val 600 --base_lr 0.002 --gpu_id 2
```

### 断点恢复训练
将yaml文件中base下的restore置为True,restore_file填上恢复训练的模型地址，运行：
```shell
python3 tools/det_train.py --config ./config/det_DB_mobilev3.yaml --log_str train_log  --n_epoch 1200 --start_val 600 --base_lr 0.002 --gpu_id 2
```

## Django website

```shell
cd mysite
python manage.py runserver
```

### 效果展示

## Document

```
```


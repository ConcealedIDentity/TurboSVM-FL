# Data preprocessing for FEMNIST dataset

## Source
https://github.com/TalwalkarLab/leaf/tree/master/data/femnist.

## Steps

Please go through the following steps in Linux Python exvironment that includes `numpy` and `pillow` packages. If running in Windows Subsystem Linux (WSL), please make sure `unzip` in installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

1. git clone from https://github.com/TalwalkarLab/leaf/tree/master.

2. dive into `leaf/data/femnist/`.

3. run `./preprocess.sh -s niid --sf 1.0 -k 0 -t user --tf 0.9 --smplseed 0 --spltseed 0`.

4. place the json files according to the path variables `--femnist_train_path` and `--femnist_test_path` in `utils.py`.

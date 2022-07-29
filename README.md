# split-dqn-singulation
This repository is an implementation of the paper 'Split Deep Q-Learning for Robust Object Singulation' in PyBullet.

## Installation
```shell
git clone git@github.com:robot-clutter/split-dqn-singulation.git
cd split-dqn-singulation

virtualenv ./venv --python=python3
source ./venv/bin/activate
pip install -r requirements.txt
```

Install PytTorch 1.9.0
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Download and install the core code used for clutter tasks:
```shell
git clone git@github.com:robot-clutter/clutter_core.git
cd clutter_core
pip install -e .
cd ..
```


## Quick Demo
To download the pretrained model, run the following command:
```commandline
cd downloads
./download-weights.sh
cd ..
```

This demo runs our pre-trained model with a UR5 robot arm in simulation. The objective is to singulate the target object (red one) from its surrounding clutter.
```commandline
python run.py --is_testing --test_trials 10 --episode_max_steps 10 --seed 100
```

## Training
To train the split-dqn agent in simulation run the following command:
```commandline
python run.py --n_episodes 10000 --episode_max_steps 10 --save_every 100 --seed 0
```

To train from scratch in the more complex environment using the extra primitive:
```commandline
python run.py --env complex --n_episodes 10000 --episode_max_steps 10 --save_every 100 --seed 0
```

If you want to use the pretrained models and just train the extra q-network i.e. the one corresponding to the extra primitive:
```commandline
python run.py --checkpoint checkpoint --env complex --n_episodes 10000 --episode_max_steps 10 --save_every 100 --seed 0
```

## Evaluation
To test your own trained model, simply change the location of --checkpoint:
```commandline
python run.py --is_testing --checkpoint checkpoint --test_trials 100 --episode_max_steps 10 --seed 1
```

## Citing
If you find this code useful in your work, please consider citing:
```shell
@inproceedings{sarantopoulos2020split,
  title={Split deep q-learning for robust object singulation},
  author={Sarantopoulos, Iason and Kiatos, Marios and Doulgeri, Zoe and Malassiotis, Sotiris},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6225--6231},
  year={2020},
  organization={IEEE}
}
```
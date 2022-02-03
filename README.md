## Runnable Baselines

This repo contains a runnable [openai/baseline](https://github.com/openai/baselines) that uses Poetry as the package manager.

## Prerequisites 

Prerequisites:
* Python >=3.8,<3.9
* [Poetry](https://python-poetry.org)
* Cuda 11

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

# mujoco dependencies
apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
```

You may also need to set up poetry to use the correct python like this:

```
poetry env use /home/test/.pyenv/versions/3.8.11/bin/python
```

## Installation

```bash
poetry install
poetry install -E mujoco
poetry install -E atari
pip install nvidia-tensorflow
pip install nvidia-tensorboard==1.15
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```

### Example 2. DQN on Atari 
DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:
```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

### Example 3. PPO with Atari

Run the following script.

```bash
OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4
```

Open a new terminal and run
```
tensorboard --logdir runs
```


### Example 4. Experiment tracking

```bash
OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --network cnn_lstm --num_env 8 --track

OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --network cnn_lstm --num_env 8 --track --num_timesteps=10000000


python -m baselines.run --alg=ppo2 --env=CartPole-v1 --network lstm --num_env 8 --track

python -m baselines.run --alg=ppo2 --env=CartPole-v1 --network lstm --num_env 8 --nsteps 128 --nminibatches 4

```
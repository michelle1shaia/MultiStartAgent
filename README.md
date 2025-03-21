Prerequisites:
* Python >=3.7.1,<3.11
* [Poetry 1.2.1+](https://python-poetry.org)

To run experiments locally, give the following a try:

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install

# alternatively, you could use `poetry shell` and do
# `python run cleanrl/ppo.py`
poetry run python cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000

# open another terminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
```

To use experiment tracking with wandb, run
```bash
wandb login # only required for the first time
poetry run python cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000 \
    --track \
    --wandb-project-name cleanrltest
```
### Train pytorch
Install repo
`https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail`

Don't forget to install repo `pytorch-a2c-ppo-acktr-gail` with `python setup.py install`

### Train pytorch

```python tutorials/pytorch_train.py --env-name "lab_env1" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01```
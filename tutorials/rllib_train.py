import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from flow.envs.nemodrive_lab.build_envs import make_lab_env

env_name = "lab_env2"
gym_name, flow_params, flow_json = make_lab_env(env_name)

# number of parallel workers
N_CPUS = 4
N_GPUS = 1
# number of rollouts per training iteration
N_ROLLOUTS = 4

ray.init(num_cpus=N_CPUS, num_gpus=N_GPUS)

alg_run = "PPO"
HORIZON = 128

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
config["sample_batch_size"] = HORIZON
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [128, 128, 128, 128]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97
config["sgd_minibatch_size"] = min(HORIZON, config["train_batch_size"])  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 500  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon
config["num_gpus"] = 1
config["lr"] = 2.5e-4

config["vf_loss_coeff"] = 0.5
config["entropy_coeff"] = 0.01
config["clip_param"] = 0.1

config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 1,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": 500,  # number of iterations to stop after
        },
    },
})


# Tesnroforce, TF-Agents, RLLib, stable-baselines, ppo-acktr
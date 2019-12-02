from flow.envs.nemodrive_lab import LAB_ENVS
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
import json


def make_lab_env(env_name, sim_step=0.1, render=False, emission_path='data',
                 restart_instance=True, print_warnings=False):

    assert env_name in LAB_ENVS, f"{env_name} not in LAB_ENVS"

    ENV = LAB_ENVS[env_name]

    network_name = ENV["NETWORK"]
    name = env_name
    vehicles = ENV["VEHICLES"]()
    net_params = NetParams(additional_params=ENV["ADDITIONAL_NET_PARAMS"])
    initial_config_param = ENV["INITIAL_CONFIG_PARAMS"]

    initial_config = InitialConfig(**initial_config_param)

    env_name = ENV["ENVIRONMENT"]
    add_env_params = ENV["ADDITIONAL_ENV_PARAMS"]
    add_env_params["action_space_box"] = True
    env_params = EnvParams(additional_params=add_env_params, horizon=ENV["HORIZON"])

    sumo_params = SumoParams(sim_step=sim_step, render=render, emission_path=emission_path,
                             restart_instance=restart_instance, print_warnings=print_warnings)

    flow_params = dict(
        # name of the experiment
        exp_tag=name,
        # name of the flow environment the experiment is running on
        env_name=env_name,
        # name of the network class the experiment uses
        network=network_name,
        # simulator that is used by the experiment
        simulator='traci',
        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=sumo_params,
        # environment related parameters (see flow.core.params.EnvParams)
        env=env_params,
        # network-related parameters (see flow.core.params.NetParams and
        # the network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net_params,
        # vehicles to be placed in the network at the start of a rollout
        # (see flow.core.vehicles.Vehicles)
        veh=vehicles,
        # (optional) parameters affecting the positioning of vehicles upon
        # initialization/reset (see flow.core.params.InitialConfig)
        initial=initial_config
    )
    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)  # generating a string version of flow_params

    create_env, gym_name = make_create_env(params=flow_params, version=0)
    create_env()
    return gym_name, flow_json

import numpy as np
import pickle

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


from rlbench.tasks import MT15_V1, MT30_V1, MT55_V1, MT100_V1


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=5)
args = parser.parse_args()

def convert_to_snake_case(input_string):
    result = [input_string[0].lower()]  # Start with the first character in lowercase

    for char in input_string[1:]:
        if char.isupper():
            result.extend(['_', char.lower()])
        else:
            result.append(char)

    return ''.join(result)


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env.launch()

train_tasks = MT100_V1["train"]

for task in train_tasks:
    task = env.get_task(task)

    demos = task.get_demos(2, live_demos=live_demos)  # -> List[List[Observation]]
    
    name = task.get_name()

    print('Done with {}'.format(name))

    with open("demos/{}_demo_lan_{}_pertask.pkl".format(name, args.num_demos), "wb") as f:
        pickle.dump(demos, f)
    
print("Done generating combo {}".format(name))
print("=====================================")

env.shutdown()

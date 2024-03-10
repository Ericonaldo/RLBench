import numpy as np
import pickle
import pathlib

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK
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


from transformations import quaternion_multiply, quaternion_conjugate

def calculate_relative_pose(pose1, pose2):
    """
    Calculate the relative pose between two poses in the world frame.

    Args:
    - pose1 (list): List containing the [x, y, z, qx, qy, qz, qw] of the first pose.
    - pose2 (list): List containing the [x, y, z, qx, qy, qz, qw] of the second pose.

    Returns:
    - relative_pose (list): List containing the [delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, delta_qw]
      representing the relative pose.
    """
    # Extract translation and rotation components from the poses
    x1, y1, z1, qx1, qy1, qz1, qw1 = pose1
    x2, y2, z2, qx2, qy2, qz2, qw2 = pose2

    # Calculate relative translation
    delta_x = x2 - x1
    delta_y = y2 - y1
    delta_z = z2 - z1

    # Calculate relative rotation using quaternion operations
    q1 = np.array([qx1, qy1, qz1, qw1])
    q2_conjugate = quaternion_conjugate(np.array([qx2, qy2, qz2, qw2]))
    relative_rotation = quaternion_multiply(q2_conjugate, q1)

    # Extract the relative quaternion components
    delta_qx, delta_qy, delta_qz, delta_qw = relative_rotation

    # Return the relative pose
    relative_pose = [delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, delta_qw]
    return relative_pose


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

demo_keys = ['front_depth', 'front_mask', 'front_point_cloud', 'front_rgb', 'gripper_joint_positions', \
            'gripper_matrix', 'gripper_open', 'gripper_pose', 'gripper_touch_forces', 'joint_forces', 'joint_positions', 'joint_velocities', \
            'left_shoulder_depth', 'left_shoulder_mask', 'left_shoulder_point_cloud', 'left_shoulder_rgb', 'misc', 'overhead_depth', \
            'overhead_mask', 'overhead_point_cloud', 'overhead_rgb', 'right_shoulder_depth', 'right_shoulder_mask', \
            'right_shoulder_point_cloud', 'right_shoulder_rgb', 'task_low_dim_state', 'wrist_depth', 'wrist_mask', 'wrist_point_cloud', 'wrist_rgb']

train_tasks = MT100_V1["train"]

anno_dict = {"episode":{"length":[]}, "language":{"task":[], "instruction":[]}}

p = pathlib.Path("demos")
p.mkdir(parents=True, exist_ok=True)

total_num = 0
for task in train_tasks:
    task = env.get_task(task)

    episode = dict({"actions":[]})
    demos = task.get_demos(2, live_demos=live_demos)  # -> List[List[Observation]]
    name = task.get_name()
    
    for j,traj in enumerate(demos):
        traj, lang = traj
        anno_dict["episode"]["length"].append(len(traj))
        anno_dict["language"]["task"].append(name)
        anno_dict["language"]["instruction"].append(lang)
        for i,step in enumerate(traj[:-1]):
            for key in demo_keys:
                if key not in episode:
                    episode[key] = []
                episode[key].append(getattr(step, key))
            action = calculate_relative_pose(task._robot.arm.get_tip().get_pose(), traj[i+1].gripper_pose) # The next frame gripper pose relative to the current gripper
            action.append(step.gripper_open)
            episode["actions"].append(action)
            
        with open("demos/episode_{}.pkl".format(total_num), "wb") as f:
            pickle.dump(demos, f)
    
        print('Done with {}, episode {}'.format(name, j))
    print("=====================================")
    print("Done generating task {}".format(name))
    print("=====================================")
        
    total_num += 1

with open("anno.npy", "wb") as f:
    np.save(f, anno_dict)

env.shutdown()

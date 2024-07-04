# utils
import copy
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import cv2 as cv
import numpy as np
import yaml

sys.path.append(f"/home/{os.getlogin()}/isaac_sim_ws/devel/lib/python3/dist-packages")

# ROS
import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose, TwistStamped
from geometry_msgs.msg import PoseStamped, Quaternion
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import message_filters
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState

pi_2 = math.pi / 2


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class TrajectoryRecorder:
    def __init__(self, dataset_root_path, trajectory_num, height, width, resolution):
        # To reset Simulations
        rospy.logdebug("START init trajectory recorder")
        self.dataset_root_path = dataset_root_path

        # pool
        self.trajectory_num = trajectory_num
        self.height = height
        self.width = width
        self.resolution = resolution
        self.step_num = 0
        self.laser_dataset_pool = []
        self.global_path_pool = []
        self.local_path_pool = []
        self.dataset_info = {}

        # write threading
        self.dataset_thread = threading.Thread(target=self._writing_thread)
        self.dataset_condition_lock = threading.Condition()
        self.close_signal = False

        # ros communication
        self.scan_sub = message_filters.Subscriber("/front/scan", LaserScan)
        self.cmd_vel_sub = message_filters.Subscriber("/cmd_vel_stamped", TwistStamped)
        self.path_sub = message_filters.Subscriber("/move_base/GlobalPlanner/robot_frame_plan", Path)
        self.local_path_sub = message_filters.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path)
        subs = [self.scan_sub, self.cmd_vel_sub, self.path_sub, self.local_path_sub]
        # self.msg_filter = message_filters.TimeSynchronizer(subs, 2000)
        self.msg_filter = message_filters.ApproximateTimeSynchronizer(subs, queue_size=1000, slop=0.01)
        self.msg_filter.registerCallback(self._sensor_callback)

        self.nav_client = SimpleActionClient("move_base", MoveBaseAction)
        self.reset_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.state_client = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.target_pose = Pose()
        self.init_pose = (-width / 2, 3.0)
        self.goal_pose = (-width / 2, 3.0 + height + 3.0)

        # flags
        self.counter = 0
        self.result = False
        self.reset()
        self.dataset_thread.start()
        rospy.logfatal("Finished Init trajectory recorder")

    def cycle(self):
        curr_time = time.time()
        start_time = time.time()
        robot_pose = self.state_client("jackal", "world")
        while robot_pose.pose.position.y < self.height + 1.0 and curr_time - start_time < 100.0:
            curr_time = time.time()
            robot_pose = self.state_client("jackal", "world")
            curr_pose = (robot_pose.pose.position.x, robot_pose.pose.position.y)
            print(f"Time: {curr_time - start_time}(s), x: {curr_pose[0]}(m), "
                  f"y: {curr_pose[1]}(m), world_height:{self.height + 1}(m)", end="\r")
            while time.time() - curr_time < 0.1:
                rospy.sleep(0.01)
        rospy.logfatal(curr_time - start_time)
        self.result = curr_time - start_time < 100.0
        return self.result

    def _writing_thread(self):
        while not self.close_signal:
            self.dataset_condition_lock.acquire()
            if len(self.laser_dataset_pool) == 0:
                self.dataset_condition_lock.wait()
            laser_dataset = copy.deepcopy(self.laser_dataset_pool)
            global_plan_dataset = copy.deepcopy(self.global_path_pool)
            local_plan_dataset = copy.deepcopy(self.local_path_pool)
            self.laser_dataset_pool.clear()
            self.global_path_pool.clear()
            self.local_path_pool.clear()
            self.dataset_condition_lock.release()
            for i in range(len(laser_dataset)):
                laser, laser_path = laser_dataset[i]
                global_plan, global_plan_path = global_plan_dataset[i]
                local_plan, local_plan_path = local_plan_dataset[i]
                np.save(laser_path, np.array(laser))
                self._save_plan(global_plan, global_plan_path)
                self._save_plan(local_plan, local_plan_path)
            del laser_dataset
            del global_plan_dataset
            del local_plan_dataset

    def _save_plan(self, plan: Path, path: str):
        poses = []
        for pose in plan.poses:
            assert isinstance(pose, PoseStamped)
            poses.append([
                pose.pose.position.x,
                pose.pose.position.y,
                self._get_yaw(pose.pose.orientation)
            ])
        np.save(path, np.array(poses))

    def _sensor_callback(self, scan_msg: LaserScan, cmd_vel_msg: TwistStamped, path_msg: Path, local_path_msg: Path):
        self.counter += 1
        laser_path = os.path.join(f"{self.dataset_root_path}/trajectory{self.trajectory_num}",
                                  f"laser{self.step_num}.npy")
        global_plan_path = os.path.join(
            f"{self.dataset_root_path}/trajectory{self.trajectory_num}",
            f"global_plan{self.step_num}.npy")
        local_plan_path = os.path.join(
            f"{self.dataset_root_path}/trajectory{self.trajectory_num}",
            f"local_plan{self.step_num}.npy")

        robot_state = self.state_client("jackal", "world")

        data = {
            "time": rospy.Time.now().to_sec(),
            "robot_x": robot_state.pose.position.x,
            "robot_y": robot_state.pose.position.y,
            "robot_yaw": self._get_yaw(robot_state.pose.orientation),
            "cmd_vel_linear": cmd_vel_msg.twist.linear.x,
            "cmd_vel_angular": cmd_vel_msg.twist.angular.z,
            "laser_path": laser_path,
            "global_plan_path": global_plan_path
        }
        self.dataset_info["data"].append(data)
        self.dataset_condition_lock.acquire()
        self.laser_dataset_pool.append((scan_msg.ranges, laser_path))
        self.global_path_pool.append((path_msg, global_plan_path))
        self.local_path_pool.append((local_path_msg, local_plan_path))
        self.dataset_condition_lock.notify()
        self.dataset_condition_lock.release()
        self.step_num += 1

    def reset(self):
        rospy.logfatal("Start initializing robot...")
        self.step_num = 0

        if os.path.exists(f"{self.dataset_root_path}/trajectory{self.trajectory_num}"):
            os.system(f"rm -rf {self.dataset_root_path}/trajectory{self.trajectory_num}")
        os.mkdir(f"{self.dataset_root_path}/trajectory{self.trajectory_num}")

        self.dataset_info = {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "data": []
        }
        self.target_pose.position.x = self.height + 3.0
        self.target_pose.position.y = 0.0
        self.target_pose.orientation = Quaternion(0, 0, np.sin(pi_2 / 2.), np.cos(pi_2 / 2.))
        self._publish_goal_position(self.target_pose)

    def close(self):
        info_file = os.path.join(f"{self.dataset_root_path}/trajectory{self.trajectory_num}", "dataset_info.json")
        with open(info_file, "w") as f:
            json.dump(self.dataset_info, fp=f)
        self.dataset_condition_lock.acquire()
        self.close_signal = True
        self.dataset_condition_lock.notify()
        self.dataset_condition_lock.release()
        self.dataset_thread.join()
        print("=" * 50)
        print(f"counter:{self.counter}")
        print("=" * 50)

    def _publish_goal_position(self, pose: Pose):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "base_link"
        goal.target_pose.pose = pose
        self.nav_client.wait_for_server()
        self.nav_client.send_goal(goal)

    @staticmethod
    def _get_yaw(quaternion: Quaternion):
        _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return yaw


def reset_gazebo_pose(world_yaml, world_pgm):
    reset_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    state_client = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    reset_client.wait_for_service()
    state_client.wait_for_service()
    with open(world_yaml, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    resolution = config["resolution"]
    pgm_file = cv.imread(world_pgm, flags=cv.IMREAD_GRAYSCALE)
    height, width = pgm_file.shape
    height *= resolution
    width *= resolution

    # reset robot pose in gazebo
    reset_pose = ModelState()
    reset_pose.model_name = 'jackal'
    reset_pose.pose.position.x = -width / 2
    reset_pose.pose.position.y = 3.0
    reset_pose.pose.position.z = 0.0
    reset_pose.pose.orientation = Quaternion(0, 0, np.sin(pi_2 / 2.), np.cos(pi_2 / 2.))
    rospy.wait_for_service("/gazebo/set_model_state")
    init_pose = (-width / 2, 3.0)
    robot_pose = state_client("jackal", "world")
    curr_pose = (robot_pose.pose.position.x, robot_pose.pose.position.y)
    while compute_distance(init_pose, curr_pose) > 0.1:
        reset_client(reset_pose)
        robot_pose = state_client("jackal", "world")
        curr_pose = (robot_pose.pose.position.x, robot_pose.pose.position.y)
        time.sleep(1.0)
    return height, width, resolution


def main(dataset_root_path, trajectory_num, world_yaml, world_pgm):
    rospy.init_node("trajectory_recorder")
    height, width, resolution = reset_gazebo_pose(world_yaml, world_pgm)
    recorder = TrajectoryRecorder(dataset_root_path, trajectory_num, height, width, resolution)
    result = recorder.cycle()
    recorder.close()
    del recorder
    with open(f"{dataset_root_path}/trajectory{trajectory_num}/result.txt", "w") as f:
        f.write(str(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to recorder data from gazebo")
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--root", type=str, default=f"/home/{os.getlogin()}/Downloads/dataset")
    parser.add_argument("--map_yaml", type=str, required=True)
    parser.add_argument("--map_pgm", type=str, required=True)
    args = parser.parse_args()
    print(args.root)
    main(args.root, args.num, args.map_yaml, args.map_pgm)


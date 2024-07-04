import subprocess
import time

import recorder
import os
import rospkg

max_test_times = 10

dataset_root_path = f"/home/{os.getlogin()}/Downloads/dataset"

if __name__ == "__main__":
    ros_package = rospkg.RosPack()
    base_path = ros_package.get_path('trajectory_recorder')
    length = len(os.listdir(os.path.join(base_path, "test_data/world_files")))
    for i in range(951, length):
        map_config_file = os.path.join(base_path, f"test_data/map_files/yaml_{i}.yaml")
        map_pgm_file = os.path.join(base_path, f"test_data/map_files/map_pgm_{i}.pgm")
        world_file = os.path.join(base_path, f"test_data/world_files/world_{i}.world")
        for j in range(max_test_times):
            print("=" * 50)
            print(f"World{i} try for the {j + 1} times")
            print("=" * 50)
            # roscore
            core_process = subprocess.Popen(
                "roscore"
            )
            time.sleep(2.0)
            # gazebo
            gazebo_process = subprocess.Popen(
                ["roslaunch", "jackal_helper", "gazebo_launch.launch", f"world_name:={world_file}", "gui:=true"]
            )
            time.sleep(3.0)
            # navigation
            navigation_process = subprocess.Popen(
                ["roslaunch", "jackal_helper", "move_base_teb.launch"]
            )
            time.sleep(2.0)
            # recorder
            recorder_process = subprocess.Popen(
                ["python", f"/home/{os.getlogin()}/isaac_sim_ws/src/trajectory_recorder/scripts/recorder.py",
                 "--num", str(i),
                 "--root", dataset_root_path,
                 "--map_yaml", map_config_file,
                 "--map_pgm", map_pgm_file]
            )
            # wait for result
            recorder_process.wait()
            navigation_process.terminate()
            navigation_process.wait()
            gazebo_process.terminate()
            gazebo_process.wait()
            core_process.terminate()
            core_process.wait()
            with open(f"{dataset_root_path}/trajectory{i}/result.txt", "r") as f:
                result = eval(f.readline())
            if result is True:
                print("record done! try next world!")
                break
            else:
                print("navigation overtime, try again")

# mountain_road_donkey_car

Reporitory for evaluating different classic constrol and DL methods for self-driving rc-cars.

# Installation
1. ```git clone```
2. ```cd mountain_road_donkey_car && git submodule update --init --recursive```
if you wish to run all in docker run below. Nvidia-gpu container should be installed.
3. ```docker build -f docker/Dockerfile -t rc-car-runner .```
4. Or download simulator from ```"https://github.com/tawnkramer/gym-donkeycar/releases/download/v2020.5.16/DonkeySimLinux.zip"```

# Training
1. Collect drive samples from Unity simulator or download this test ones


# Running

1. ```python rc_car/run.py --sim-path=/home/robot/DonkeySimLinux/donkey_sim.x86_64 --save-logs=True --model=tb_logs/linear_clean/linear_clean.pt --model-type=cnn```

or in docker

2. ```docker run -it --gpus 'all,"capabilities=graphics,compute,utility,video"' -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v "$(pwd)/data:/data" -v "$(pwd)/tb_logs:/workspace/tb_logs" --rm --name mountain_car --ipc=host jb-assigment /bin/bash -c "python /car/rc_car/run.py --sim-path="/car/DonkeySimLinux/donkey_sim.x86_64" --save-logs=True --model=tb_logs/linear_without_aug/linear_without_aug.pt --model-type=cnn"```

# Currently supported sumulators
1. Donkey-Simulator

# Currently supported models
1. CNN from nvidia end-to-end paper


# RC Car deep learning library

![Docker Image CI](https://github.com/AlexKaravaev/mountain_road_donkey_car/workflows/Docker%20Image%20CI/badge.svg)
![Pytest](https://github.com/AlexKaravaev/mountain_road_donkey_car/workflows/Pytest/badge.svg)

![Alt text](media/car.png?raw=true "Self driving RC car")

Reporitory for evaluating different classic constrol and DL methods for self-driving rc-cars.

List of currently supported models and simulators

*Simulators*
* DonkeyGym [github](https://github.com/tawnkramer/gym-donkeycar)

*Models*
* Nvidia end-to-end driving model [arxiv](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
## Installation
* ```git clone```
* ```cd mountain_road_donkey_car && git submodule update --init --recursive```
if you wish to run all in docker run below. Nvidia-gpu container should be installed.
* ```docker build -f docker/Dockerfile -t rc-car-runner .```
* Or download simulator from ```"https://github.com/tawnkramer/gym-donkeycar/releases/download/v2020.5.16/DonkeySimLinux.zip"```

## Training
* Collect drive samples from Unity simulator or download this test [dataset]("https://www.dropbox.com/s/h2lkl44zlgu9804/data.zip?dl=0") and unzip it to data folder in current dir.

* run ```python ./rc_car/train.py --path-to-training-data data/driving/ data/driving_2 data/driving_3 data/driving_4 --model-type=cnn --model=linear_shuffle --hyperparam-file=rc_car/train_conf.json ```

or in docker

* ```docker run -it --gpus all -v "$(pwd)/data:/data" -v "$(pwd)/tb_logs:/workspace/tb_logs" -v "$(pwd)/rc_car/train_conf.json:/train_conf.json" --rm --name mountain_car --ipc=host rc-car /bin/bash -c "python /car/rc_car/train.py --path-to-training-data data/driving/ data/driving_2 data/driving_3 data/driving_4 --model-type=cnn --model=linear_shuffle --hyperparam-file=rc_car/train_conf.json"```


Training logs can be accessed via tensorboard
```
tensorboard --logdir=tb_logs
```

## Running

* Train or download sample weights from [weights](https://www.dropbox.com/s/aqanco2ji308prf/linear_shuffle.pt?dl=0)

* ```python rc_car/run.py --sim-path=/home/robot/DonkeySimLinux/donkey_sim.x86_64 --save-logs=True --model=tb_logs/linear_clean/linear_clean.pt --model-type=cnn```

or in docker

* ```docker run -it --gpus 'all,"capabilities=graphics,compute,utility,video"' -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v "$(pwd)/data:/data" -v "$(pwd)/tb_logs:/workspace/tb_logs" --rm --name mountain_car --ipc=host jb-assigment /bin/bash -c "python /car/rc_car/run.py --sim-path="/car/DonkeySimLinux/donkey_sim.x86_64" --save-logs=True --model=tb_logs/linear_without_aug/linear_without_aug.pt --model-type=cnn"```

## Viewing tracks 
After model ran on particular track, logs from that model are saved into ./logs/ directory. You can view them via 
```
python rc_car/plot/plotting.py --log-file=./logs/recorded_data/path_to_logfile.json
```


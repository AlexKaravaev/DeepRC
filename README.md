# RC Car deep learning library

![Docker Image CI](https://github.com/AlexKaravaev/mountain_road_donkey_car/workflows/Docker%20Image%20CI/badge.svg)
![Pytest](https://github.com/AlexKaravaev/mountain_road_donkey_car/workflows/Pytest/badge.svg)

![RC Car](media/car.png?raw=true "Self driving RC car")

Reporitory for evaluating different classic constrol and DL methods for self-driving rc-cars.

List of currently supported models and simulators

*Simulators*
* DonkeyGym [github](https://github.com/tawnkramer/gym-donkeycar)

*Models*
* Nvidia end-to-end driving model [arxiv](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

Complete instructions and list of supported models are in project [wiki](https://github.com/AlexKaravaev/DeepRC/wiki)
## Installation
* ```git clone```
* ```cd DeepRC && git submodule update --init --recursive```

if you wish to run all in docker run below. Nvidia-gpu container should be installed.
* ```docker build -f docker/Dockerfile -t rc-car-runner .```
* Or download simulator from ```"https://github.com/tawnkramer/gym-donkeycar/releases/download/v2020.5.16/DonkeySimLinux.zip"``` and build package locally via ```python setup.py install``` with installing all requirements


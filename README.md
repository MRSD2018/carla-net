# carla-net

# Title TO DO:

### Introduction
Generating datasets from real life environment is a vey challenging task , as there are a lot of variables involved which not only makes the process difficult but also consumes a lot of time and effort. To mitigate this and to accelerate the process of generating larger amounts of datsets in a shorter span of time, we here use a simulator called 'Carla', which is an open-source simulator for autonomous driving research. To know more about this please visit their website [Carla Simulator](http://carla.org/)
```
![Alt text](https://github.com/MRSD2018/carla-net/tree/icnet-init/Results/carla-black-m.png ![enter image description here](https://github.com/MRSD2018/carla-net/tree/icnet-init/Results/carla-black-m.png)"Optional title")
```
Leveraging the power of Carla to generate real enivronment scenes , we generated about 50 Gb of high qualitydata,where the envioronment had different weather conditions.


### Problem we are solving
We use the dataset generated from Carla to perform transfer learning on an pre-trained network learned on citsycapes dataset and  then we validate the performance on data from real world. 

### Approach
IC-net which is now state-of-the art in real time semantic segmetation was used to train on the dataset generated from the simulator.
### Installation
- Install deep learning framework -Pytorch
- Install Carla simulator
- 
### Dataset
We used Cityscape data for this project. You can download it by going to this link
[CityScape](https://www.cityscapes-dataset.com/).

### Steps to be followed
- 

### Results
We see that the network is able to learn and perfrom reasonably well in the real world data.

Finetuning imagenet on carla simulation renderings

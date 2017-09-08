# HisDB
My submission for the ICDAR2017 Competition on Layout Analysis for Challenging Medieval Manuscripts [(HISDB)](https://diuf.unifr.ch/main/hisdoc/icdar2017-hisdoc-layout-comp).  See the competition web site for precise input/output formats.

We have a docker image to make it easy to run our models on new data.  You must have the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin installed to use it though you can still run our models on CPU (not recommended).

The usage for the docker container is

```
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:hisdb_gpu python task_1/task1.py /data/input_file.jpg /data/output_file.png $DEVICE_ID
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:hisdb_gpu python task_2/task2.py /data/input_file.jpg /data/input_file.xml /data/output_file.xml $DEVICE_ID
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:hisdb_gpu python task_3/task3.py /data/input_file.jpg /data/input_file.xml /data/output_file.xml $DEVICE_ID
```

`$HOST_WORK_DIRECTORY` is a directory on your machine that is mounted on /data inside of the docker container (using -v).  It's the only way to expose images to the docker container.
`$DEVICE_ID` is the ID of the GPU you want to use (typically 0).  If omitted, then the models are run in CPU mode.
There is no need to download the containers ahead of time.  If you have docker and nvidia-docker installed, running the above commands will pull the docker image (~2GB) if it has not been previously pulled.

#!/bin/bash
if [ ! -d f1tenth_gym ] ; then
    git clone https://github.com/cosynus-lix/f1tenth_gym
    cd f1tenth_gym
    git checkout cpp_backend_archive_add_obstacle
    cd ../
else
    echo f1tenth_gym exists, not cloning.
fi
docker build -t f1tenth_gym -f Dockerfile .

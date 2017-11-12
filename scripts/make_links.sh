#!/bin/bash

ln -Lrsf ../../dataset/samplers.py dataset
ln -Lrsf ../../dataset/replay_buffer.py dataset
ln -Lrsf ../../environ/environment.py environ
ln -Lrsf ../../environ/synth.py environ
ln -Lrsf ../../model/discriminator.py model
ln -Lrsf ../../model/generator.py model
ln -Lrsf ../../model/utils.py model
ln -Lrsf ../../main.py .

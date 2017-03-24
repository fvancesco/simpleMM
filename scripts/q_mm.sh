#!/bin/bash
#$ -N mmsergio
#$ -cwd
#$ -e $HOME/logs/$JOB_NAME-$JOB_ID.err
#$ -o $HOME/logs/$JOB_NAME-$JOB_ID.out
#$ -l h=node01|node02|node04|node05|node06|node07|node08|node09|node10
#$ -l gpu=1
#$ -q default.q

module load cuda/7.5

########
cd /homedtic/fbarbieri/
. /homedtic/fbarbieri/torch/install/bin/torch-activate

export LD_LIBRARY_PATH="/soft/openblas/openblas-0.2.18/lib:/homedtic/fbarbieri/libraries/hdf5-1.8.17/lib:/homedtic/fbarbieri/libraries/cudnn/cuda/lib64:$LD_LIBRARY_PATH:/homedtic/fbarbieri/torch/install/lib/luarocks/rocks/image/1.1.alpha-0/lib/"
export CUDA_TOOLKIT_ROOT_DIR="/soft/cuda/cudnn/cuda/lib64:/soft/cuda/cudnn/cuda/include"
export PROTOBUF_LIBRARY="/usr/lib"
export PROTOBUF_INCLUDE_DIR="/usr/include/google/protobuf"

export LD_PRELOAD="/homedtic/fbarbieri/torch/install/lib/luarocks/rocks/image/1.1.alpha-0/lib/"
########

cd /homedtic/fbarbieri/git/simpleMM

#mm twitter
th train_sergio.lua -input_code 434 > /homedtic/fbarbieri/logs/mmsergio434 &
#th train_sergio.lua -input_code 664 > /homedtic/fbarbieri/logs/mmsergio664 &

#th classify_imagemusic.lua ../../../imagemusic/cp/0001/model_best.t7 ../../../imagemusic/dataset/finetune/genres/test.ls > /homedtic/fbarbieri/imagemusic/results/classifiy_genres_test.txt &
#wait
#th classify_imagemusic.lua ../../../imagemusic/cp/0001_2/model_best.t7 ../../../imagemusic/dataset/finetune2/genres/val.ls > /homedtic/fbarbieri/imagemusic/results/classifiy_genres.txt &
#wait
#th classify_imagemusic.lua ../../../imagemusic/cp/0001_2/model_best.t7 ../../../imagemusic/dataset/finetune2/genres/train.ls > /homedtic/fbarbieri/imagemusic/results/classifiy_genres.txt &


#th classify.lua ../../../imagemusic/models/resnet-18.t7 ../../../imagemusic/Cute-Animals-22.jpg > log.tmp
#th train_hpc.lua -val_size 2000 -hidden_size 9000 -print_start 15 -save_output 990 > ~/mmtwitter_old/logs/relu_pre_$(date +%Y_%m_%d_%H_%M_%S) &

wait
echo "Stai senza pensieri - classify"



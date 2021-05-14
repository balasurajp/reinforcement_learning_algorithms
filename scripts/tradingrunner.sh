#!/usr/bin/env bash

command=$1
dataconfig=$2
interpreter="~/suraj/envs/py36/bin/activate"

source ~/suraj/envs/py36/bin/activate
pip install -e ./

if [ "${command}" == "run" ]; then
    screen -S PPO_PO -t win1 -A -d -m
    screen -S DDPG_PO -t win1 -A -d -m
    screen -S SAC_PO -t win1 -A -d -m

    screen -S PPO_PO -p win1 -X stuff $"source ${interpreter} ;"
    screen -S DDPG_PO -p win1 -X stuff $"source ${interpreter} ;"
    screen -S SAC_PO -p win1 -X stuff $"source ${interpreter} ;"

    screen -S PPO_PO -p win1 -X stuff $"rlclustertrain ppo --envid ${dataconfig} --batch_size 64 --maxsteps_per_iteration 2000 --num_iterations 250 --output_layer dirichlet ;"
    screen -S DDPG_PO -p win1 -X stuff $"rlclustertrain ddpg --envid ${dataconfig} --batch_size 64 --actor_lrp 2e-4 --critic_lrq 1e-3 --num_iterations 250 --output_layer softmax --minsteps_per_iteration 2000 --warmup_steps 0 ;"
    screen -S SAC_PO -p win1 -X stuff $"rlclustertrain sac --envid ${dataconfig} --batch_size 64 --num_iterations 250 --output_layer dirichlet --minsteps_per_iteration 2000 --warmup_steps 0 ;"

    screen -S PPO_PO -p win1 -X stuff $"\n"
    screen -S DDPG_PO -p win1 -X stuff $"\n"
    screen -S SAC_PO -p win1 -X stuff $"\n"
fi

if [ "${command}" == "kill" ]; then
    screen -S PPO_PO -X quit
    screen -S DDPG_PO -X quit
    screen -S SAC_PO -X quit
fi
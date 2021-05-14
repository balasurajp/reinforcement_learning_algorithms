#!/usr/bin/env bash

interpreter="~/suraj/envs/py36/bin/activate"

algs=(VPG A2C TRPO PPO)
envs=(MountainCarContinuous-v0 Pendulum-v0 BipedalWalker-v3 LunarLanderContinuous-v2)
max_iter=(400 1000 1000 1000)
seeds=8

screen -S betatrails -t win1 -A -d -m
screen -S betatrails -p win1 -X stuff $"source ${interpreter} ;"
screen -S gaussiantrails -t win1 -A -d -m
screen -S gaussiantrails -p win1 -X stuff $"source ${interpreter} ;"

for (( j = 1; j <= $seeds; ++j )); do
    for (( i = 0; i < ${#envs[@]}; ++i )); do

        for (( k = 0; k < ${#algs[@]}; ++k )); do
            screen -S betatrails -p win1 -X stuff $"python -m rlcluster.agents.${algs[$k]}.train --env_id ${envs[$i]} --dist BETA --max_iter ${max_iter[$i]} --model_path ./results/tmodels/ --log_path ./results/logs/ --seed $j --num_process 1 ;"
            screen -S gaussiantrails -p win1 -X stuff $"python -m rlcluster.agents.${algs[$k]}.train --env_id ${envs[$i]} --dist GAUSSIAN --max_iter ${max_iter[$i]} --model_path ./results/tmodels/ --log_path ./results/logs/ --seed $j --num_process 1 ;"
        done

    done
done

screen -S betatrails -p win1 -X stuff $"\n"
screen -S gaussiantrails -p win1 -X stuff $"\n"

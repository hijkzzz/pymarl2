#!/bin/bash
trap 'onCtrlC' INT
set -x
function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done
  
  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

config=$1  # qmix
tag=$2
maps=${3:-sc2_gen_protoss_epo,sc2_gen_terran_epo,sc2_gen_zerg_epo}   # MMM2 left out
units=${8:-6}
offset=0
threads=${4:-24} # 2
td_lambdas=${9:-0.6}
eps_anneals=${10:-100000}
args=${5:-}    # ""
gpus=${6:-0,1,2,3,4,5,6,7}    # 0,1
times=${7:-3}   # 5
prob_obs_enemy=${11:-0.0,0.5,1.0}

maps=(${maps//,/ })
units=(${units//,/ })
enemies=(${enemies//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })
td_lambdas=(${td_lambdas//,/ })
eps_anneals=(${eps_anneals//,/ })
prob_obs_enemy=(${prob_obs_enemy//,/ })

if [ ! $config ] || [ ! $tag ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name map_name_list (experinments_threads_num arg_list gpu_list experinments_num)"
    exit 1
fi

echo "CONFIG:" $config
echo "MAP LIST:" ${maps[@]}
echo "THREADS:" $threads
echo "ARGS:"  ${args[@]}
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times
echo "TDLAMBDAS:" ${td_lambdas[@]}
echo "EPSANNEALS:" ${eps_anneals[@]}

# run parallel
count=0
for prob in "${prob_obs_enemy[@]}"; do
    for tdlambda in "${td_lambdas[@]}"; do
        for epsanneal in "${eps_anneals[@]}"; do
            for map in "${maps[@]}"; do
                for unit in "${units[@]}"; do
                    for((i=0;i<times;i++)); do
                        gpu=${gpus[$(($count % ${#gpus[@]}))]}
                        group="${config}-${map}-${tag}"
#                         enemies=$(($unit + $offset))
						enemies=5
                        ./run_docker.sh $gpu python3 src/main.py --config="$config" --env-config="$map" with group="$group" env_args.capability_config.n_units=$unit env_args.capability_config.n_enemies=$enemies env_args.prob_obs_enemy=$prob use_wandb=True td_lambda=$tdlambda epsilon_anneal_time=$epsanneal save_model=True "${args[@]}" &

                        count=$(($count + 1))
                        if [ $(($count % $threads)) -eq 0 ]; then
                            wait
                        fi
                        # for random seeds
                        sleep $((RANDOM % 3 + 3))
                    done
                done
            done
        done
    done
done
wait


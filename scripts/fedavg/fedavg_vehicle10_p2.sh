for trial in 1 2 3
do
    dir='../../save_results/vehicle10/percentage20/fedavg'
    if [ ! -e $dir ]; then
    mkdir -p $dir 
    fi 
    
    python ../../main_fedavg.py --trial=$trial \
    --rounds=300 \
    --num_users=100 \
    --frac=0.2 \
    --local_ep=10 \
    --local_bs=20 \
    --lr=0.007 \
    --momentum=0.9 \
    --model=resnet9 \
    --dataset=vehicle10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../../save_results/' \
    --partition='percentage20' \
    --alg='fedavg' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=4 \
    --print_freq=10 \
    --seed=42 \
    2>&1 | tee $dir'/'$trial'.txt'

done 

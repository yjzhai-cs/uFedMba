for trial in 1 2 3
do
    dir='../../save_results/gtsrb/percentage30/fedavg'
    if [ ! -e $dir ]; then
    mkdir -p $dir 
    fi 
    
    python ../../main_fedavg.py --trial=$trial \
    --rounds=300 \
    --num_users=100 \
    --frac=0.2 \
    --local_ep=10 \
    --local_bs=20 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=gtsrb \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../../save_results/' \
    --partition='percentage30' \
    --alg='fedavg' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=1 \
    --print_freq=10 \
    --seed=42 \
    2>&1 | tee $dir'/'$trial'.txt'

done 

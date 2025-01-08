for trial in 1 2 3
do
    dir='../../save_results/stanford_cars/percentage20/pfedaal'
    if [ ! -e $dir ]; then
    mkdir -p $dir 
    fi 
    
    python ../../main_pfedaal.py --trial=$trial \
    --rounds=300 \
    --num_users=50 \
    --frac=0.2 \
    --local_ep=10 \
    --local_bs=20 \
    --lr=0.007 \
    --momentum=0.9 \
    --model=resnet18 \
    --dataset=stanford_cars \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../../save_results/' \
    --partition='percentage20' \
    --alg='pfedaal' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    --seed=42 \
    2>&1 | tee $dir'/'$trial'.txt'

done 

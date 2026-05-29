export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

GPU_NUM=1
DATASET_ROOT='/home/carla1000/InterFuser/dataset/'

./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla --train-towns 2  --val-towns 5 \
    --train-weathers 0  --val-weathers 10 11 12 13 \
    --model interfuser_baseline --sched cosine --epochs 25 --warmup-epochs 5 --lr 0.0005 --batch-size 4  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment interfuser_baseline \
    --pretrained \
    --native-amp



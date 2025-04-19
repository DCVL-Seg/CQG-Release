GPU=0,1

CUDA_VISIBLE_DEVICES=${GPU} python train_net.py \
    --num-gpus 2 \
    --config-file configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml

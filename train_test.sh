CUDA_VISIBLE_DEVICES=3 python train.py --dataset Synapse \
--cfg 'configs/swin_tiny_patch4_window7_224_lite.yaml' \
--root_path 'datasets/Synapse' \
--max_epochs 150   \
--output_dir model_out \
--img_size 224 \
--base_lr 1e-4 \
--batch_size 24

CUDA_VISIBLE_DEVICES=3 python test.py \
--dataset Synapse \
--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
--is_saveni --output_dir model_out --max_epoch 150  \
--base_lr 1e-4 --img_size 224 --batch_size 24

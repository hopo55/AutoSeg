CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --backbone resnet --lr 0.007 --workers 4 --epochs 40 --batch_size 2 --gpu-ids 0 --eval_interval 1 --dataset sealer

export CUDA_VISIBLE_DEVICES=0

python train_vit.py --lr 1e-1 --num_epochs 5 --batch_size 16 --src_dir sports_dataset --log_dir logs --use_augmentation --save_dir models
python train_vit.py --lr 1e-3 --num_epochs 5 --batch_size 16 --src_dir sports_dataset --log_dir logs --use_augmentation --save_dir models
python train_vit.py --lr 1e-5 --num_epochs 5 --batch_size 16 --src_dir sports_dataset --log_dir logs --use_augmentation --save_dir models
python train_vit.py --lr 1e-5 --num_epochs 5 --batch_size 16 --src_dir sports_dataset --log_dir logs --save_dir models
python train_vit.py --lr 1e-5 --num_epochs 5 --batch_size 64 --src_dir sports_dataset --log_dir logs --use_augmentation --save_dir models

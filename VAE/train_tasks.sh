python train_vae.py --dir dataset/block_pyramid-vision-v0/wrist_rgb --batch_size 64 --epochs 50 --beta 1 --ngpu 4 --save_dir output/block_pyramid-vision-v0
python train_vae.py --dir dataset/open_box-vision-v0/wrist_rgb --batch_size 64 --epochs 50 --beta 1 --ngpu 4 --save_dir output/open_box-vision-v0
python train_vae.py --dir dataset/place_shape_in_shape_sorter-vision-v0/wrist_rgb --batch_size 64 --epochs 50 --beta 1 --ngpu 4 --save_dir output/place_shape_in_shape_sorter-vision-v0

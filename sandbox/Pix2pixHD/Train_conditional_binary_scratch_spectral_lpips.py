# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: eie_vision_new
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train

# %%
# Pretrain the model
# !cd ../.. ; python src/models/Pix2pixHD/train.py \
# --no_lpips_loss --ngf 64 \
# --num_D 2 --ndf 64 --n_layers_D 4 \
# --serial_batches --no_flip --niter 100 \
# --niter_decay 100 \
# --checkpoints_dir temp/checkpoint/Pix2pixHD/ \
# --name conditional_binary_spectral \
# --dataroot data/xBD_merged \
# --dataset_mode physics_aligned_bin \
# --no_instance --label_nc 0 \
# --input_nc 4 --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
# --continue_train

# %% [markdown]
# Ioannis' notes
#
# `--gpu_ids 0 \` if only 1 GPU available
#
# 'nc' means number of channels

# %%
# Check input image type & dimensions
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

img = Image.open("/media/ys/T7/eOnsight_data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/hold_A/houston_west_00000171_disaster.png")
img_tensor = transforms.ToTensor()(img)
print(f"Input image shape: {img_tensor.shape}")

img = Image.open("/media/ys/T7/eOnsight_data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west/hold_B/houston_west_00000171_disaster.png")
tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Convert to CHW
print(f"Target image shape: {tensor.shape}")


# %% [markdown]
# Your input is mask (1 channel) + RGB (3 channels) → total 4 channels.
#
# Your target is an RGB image → 3 channels.
#
#

# %% [markdown]
# Phase 1: 5 epochs with L1 only
#
#

# %% language="bash"
# cd ../.. 
# python src/models/Pix2pixHD/train.py \
#     --niter 5 --no_lsgan --no_vgg_loss --no_lpips_loss --niter_decay 0 \
#     --dataroot /media/ys/T7/eOnsight_data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west \
#     --name conditional_binary_spectral \
#     --checkpoints_dir temp/checkpoint/Pix2pixHD/ \
#     --batchSize 32 \
#     --gpu_ids 0 \
#     --input_nc 4 --label_nc 0 --output_nc 3 \
#     --loadSize 256 --fineSize 256 \
#     --nThreads 8 \
#     --save_epoch_freq 10 \
#     --no_flip \
#     --no_instance \
#     --lambda_l1 3.0 \
#     --lambda_l1_decay 0.03 \
#     --lambda_feat 10.0 \
#     --norm instance \
#     --normD spectral \
#     --netG global \
#     --ngf 32 \
#     --n_blocks_global 9 \
#     --n_local_enhancers 1 \
#     --n_blocks_local 3 \
#     --n_layers_D 4 \
#     --n_downsample_global 4 \
#     --n_downsample_E 4 \
#     --n_clusters 10 \
#     --feat_num 3 \
#     --model pix2pixHD \
#     --dataset_mode physics_aligned_bin \
#     --phase train
#

# %% [markdown]
# Phase 2: resume for more epochs with full losses
#

# %%

# !cd ../..  ;python -u src/models/Pix2pixHD/train.py \
#     --continue_train --niter 30 --niter_decay 30 \
#     --dataroot /media/ys/T7/eOnsight_data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west \
#     --name conditional_binary_spectral \
#     --checkpoints_dir temp/checkpoint/Pix2pixHD/ \
#     --batchSize 16 \
#     --gpu_ids 0 \
#     --input_nc 4 --label_nc 0 --output_nc 3 \
#     --loadSize 256 --fineSize 256 \
#     --nThreads 8 \
#     --phase train \
#     --save_epoch_freq 10 \
#     --no_flip \
#     --no_instance \
#     --lambda_l1 3.0 \
#     --lambda_l1_decay 0.03 \
#     --lambda_feat 10.0 \
#     --norm instance \
#     --normD spectral \
#     --netG global \
#     --ngf 32 \
#     --n_downsample_global 4 \
#     --n_blocks_global 9 \
#     --n_local_enhancers 1 \
#     --n_blocks_local 3 \
#     --n_layers_D 4 \
#     --n_downsample_E 4 \
#     --n_clusters 10 \
#     --feat_num 3 \
#     --model pix2pixHD \
#     --dataset_mode physics_aligned_bin


# %%
# This works
# # !cd ../.. ; python src/models/Pix2pixHD/train.py \
# #     --dataroot /media/ys/T7/eOnsight_data/eie-earth-intelligence-engine/processed/naip_to_all/houston_west \
# #     --name conditional_binary_spectral \
# #     --checkpoints_dir temp/checkpoint/Pix2pixHD/ \
# #     --batchSize 8 \
# #     --gpu_ids 0 \
# #     --input_nc 4 --label_nc 0 --output_nc 3 \
# #     --loadSize 256 --fineSize 256 \
# #     --nThreads 1 \
# #     --niter 5 --no_lsgan --no_vgg_loss --no_lpips_loss \
# #     --continue_train --niter 95 \
# #     --niter_decay 100 \
# #     --save_epoch_freq 10 \
# #     --no_flip \
# #     --no_instance \
# #     --no_lsgan \
# #     --no_vgg_loss \
# #     --no_lpips_loss \
# #     --lambda_l1 3.0 \
# #     --lambda_l1_decay 0.03 \
# #     --lambda_feat 10.0 \
# #     --norm instance \
# #     --normD spectral \
# #     --netG global \
# #     --ngf 32 \
# #     --n_blocks_global 9 \
# #     --n_local_enhancers 1 \
# #     --n_blocks_local 3 \
# #     --n_layers_D 4 \
# #     --n_downsample_global 4 \
# #     --n_downsample_E 4 \
# #     --n_clusters 10 \
# #     --feat_num 3 \
# #     --model pix2pixHD \
# #     --dataset_mode physics_aligned_bin \
# #     --phase train


# %%
# #cp pretrained model into finetuning directory
#Make a copy of the fully trained model to be finetuned on xBD_floods
# !cd ../../temp/checkpoint/Pix2pixHD/;\
# mkdir conditional_binary_spectral_lpips_finetuned ;\
# cp conditional_binary_spectral/latest* conditional_binary_spectral_lpips_finetuned/ ;\
# cp conditional_binary_spectral/iter* conditional_binary_spectral_lpips-finetuned/

# %%
# # !cd ../.. ;\
# # python src/models/Pix2pixHD/train.py \
# # --no_vgg_loss --continue_train --ngf 64 \
# # --num_D 2 --ndf 64 --n_layers_D 4  \
# # --no_flip --niter 200 --niter_decay 260 \
# # --checkpoints_dir temp/checkpoint/Pix2pixHD/ \
# # --name conditional_binary_spectral_lpips_finetuned \
# # --dataroot data/xBD_merged --dataset_mode physics_aligned_bin \
# # --no_instance --label_nc 0 --input_nc 4 --batchSize 1 --gpu_ids 0

# %%
# #Generate predictions over the test set. 
# # !cd ../.. ; python src/models/Pix2pixHD/test.py  \
# # --checkpoints_dir temp/checkpoint/Pix2pixHD/ \
# # --name conditional_binary_spectral_lpips_finetuned \
# # --phase test --how_many 2000  --dataroot data/xBD \
# # --dataset_mode physics_aligned_bin --no_instance \
# # --label_nc 0 --batchSize 8 --input_nc 4 --gpu_ids 0,1,2,3,4,5,6,7 \
# # --results_dir temp/Pix2pixHD/

# %% [markdown]
# ### We move things around at the moment in ./Temp/
# The model output is placed into ./temp/ .We use this directory to work, then we carefully selects what goes into ./results/.
# ./temp/ does not get uploaded into our bucket, only ./results/ do!

# %%
# Move synthesized results to a folder to be used by the segmentation network
# !cd ../../temp/Pix2pixHD/conditional_binary_spectral_lpips_finetuned/test_latest/ ; \
# mkdir to_segment; cp -R images/*synthesized_image.jpg to_segment

# %%
#Create Segmentation Masks
#Generate masks for all of the data generated by pix2pixHD_baseline
# ! cd ../.. ; \
# python src/models/Pix2pix-CycleGAN/test.py \
# --phase to_segment \
# --dataroot temp/Pix2pixHD/conditional_binary_spectral_lpips_finetuned/test_latest/ \
# --direction AtoB --results_dir ./temp/Pix2pixHD/conditional_binary_spectral_lpips_finetuned_masks \
# --dataset_mode single --model pix2pix \
# --checkpoints_dir ./pretrained/Pix2pix-CycleGAN/flood_segmentation/scratch_1024_plus/ \
# --name . --num_test 775 --no_flip --gpu_ids 0 \
# --max_dataset_size 2000 --batch_size 1  --load_size 1024 --crop_size 1024


# %% [markdown]
# ### Now we copy our generated images into the right ./Results/. directory

# %%
# ! cd ../.. ; \
# mkdir results/Pix2pixHD/conditional_binary_spectral_lpips_finetuned/ ; \
# cp -R temp/Pix2pixHD/conditional_binary_spectral_lpips_finetuned/test_latest/images/* results/Pix2pixHD/conditional_binary_spectral_lpips_finetuned/ ; \
# cp -R temp/Pix2pixHD/conditional_binary_spectral_lpips_finetuned_masks/to_segment_latest/images/*image_fake_B.png results/Pix2pixHD/conditional_binary_spectral_lpips_finetuned/

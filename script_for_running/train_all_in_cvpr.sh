#!/usr/bin/env bash
cd ..
bash script_for_cvpr/train_all_baseline.sh
bash script_for_cvpr/train_all_binary_bireal_layer_baseline_reparam_floatloss.sh
bash script_for_cvpr/train_all_binary_bireal_layer_baseline_reparam_gaploss.sh
bash script_for_cvpr/train_all_binary_bireal_layer_baseline_reparam_stdloss.sh
bash script_for_cvpr/train_all_binary_bireal_layer_baseline_stdloss.sh
bash script_for_cvpr/train_all_binary_bireal_layer_baseline_stdloss_scratch.sh
bash script_for_cvpr/train_all_binary_bireal_layer_best_reparam.sh
bash script_for_cvpr/train_all_binary_bireal_layer_best_reparam_stdloss.sh
bash script_for_cvpr/train_all_binary_bireal_layer_best_stdloss.sh
bash script_for_cvpr/train_all_PACS_block_index.sh
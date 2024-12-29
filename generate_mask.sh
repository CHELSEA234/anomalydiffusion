CUDA_NUM=3
path_to_mvtec_dataset="/user/guoxia11/cvl/anomaly_detection/anomaly_detection_dataset/mvtec"
CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_mask.py \
                            --data_root=$path_to_mvtec_dataset \
                            --sample_name='bottle' \
                            --anomaly_name='broken_small' \
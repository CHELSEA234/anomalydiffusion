gpu_id='7'
path_to_mvtec_dataset="/user/guoxia11/cvl/anomaly_detection/anomaly_detection_dataset/mvtec"
CUDA_VISIBLE_DEVICES=$gpu_id python run-mvtec.py --data_path=$path_to_mvtec_dataset
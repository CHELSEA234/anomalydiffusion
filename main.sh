CUDA_NUM=5
path_to_mvtec_dataset="/user/guoxia11/cvl/anomaly_detection/anomaly_detection_dataset/mvtec"
ckpt_path="/research/cvl-guoxia11/anomaly_detection_v2/AnoGen/DIFFUSION/models/ldm/text2img-large/model.ckpt"
CUDA_VISIBLE_DEVICES=$CUDA_NUM 
    python main.py \
    --spatial_encoder_embedding --data_enhance \
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml \
    -t \
    --actual_resume $ckpt_path \
    -n test \
    --gpus 0, \
    --init_word anomaly \
    --mvtec_path $path_to_mvtec_dataset

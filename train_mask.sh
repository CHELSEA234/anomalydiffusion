CUDA_NUM=1
path_to_mvtec_dataset="/user/guoxia11/cvl/anomaly_detection/anomaly_detection_dataset/mvtec"
ckpt_path="/research/cvl-guoxia11/anomaly_detection_v2/AnoGen/DIFFUSION/models/ldm/text2img-large/model.ckpt"
CUDA_VISIBLE_DEVICES=$CUDA_NUM python train_mask.py \
                            --mvtec_path=$path_to_mvtec_dataset \
                            --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
                            -t --actual_resume $ckpt_path  \
                            -n test --gpus 0, \
                            --init_word crack \
                            --sample_name='bottle' \
                            --anomaly_name='broken_small' \
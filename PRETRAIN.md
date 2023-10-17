# Pretrain

## Training FrameMAE
- After setting up your dataset, you can run the FrameMAE with the following command 
    ```bash
    MODEL_PATH="/home/srtp_ghw/fqq/MyMAE6/output_dir/checkpoint-1600.pth"
    MODEL_DEPTH="/home/srtp_ghw/fqq/output_dir/mymodel_success_really_10000_120.pth"
    python3 select_yanzheng.py \
        ${MODEL_PATH}  ${MODEL_DEPTH}

    OMP_NUM_THREADS=1   python3 -m torch.distributed.launch 
            --nproc_per_node=1 \
            --master_port 12320 --nnodes=1 \
            --node_rank=0 --master_addr=10.105.100.216 \
            run_mae_pretraining.py \
            --data_path "/home/srtp_ghw/fqq/VideoMAE/train/train.csv" \
            --img_path "/home/srtp_ghw/fqq/data/video_" \
            --mask_type tube \
            --mask_ratio 0.9 \
            --model pretrain_videomae_base_patch16_224 \
            --decoder_depth 4 \
            --batch_size 4 \
            --num_frames 16 \
            --sampling_rate 4 \
            --opt adamw \
            --opt_betas 0.9 0.95 \
            --warmup_epochs 40 \
            --save_ckpt_freq 20 \
            --epochs 41 \
            --log_dir "/home/srtp_ghw/fqq/log_dir" \
            --output_dir "/home/srtp_ghw/fqq/output_dir"
    ```
## Training Key Frmae Selector
- Put your checkpoint.py at the ideal location, then run
    ```bash
    python3 train_selecot.py \
        --mask_ratio 0.9 \
        --mask_type tube \
        --decoder_depth 4 \
        --model pretrain_videomae_base_patch16_224 \
        ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
    ```
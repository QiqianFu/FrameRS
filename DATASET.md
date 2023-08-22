# DATASET BUILD
## Build Dataset For FrameMAE

To build the dataset for FrameMAE, please download 


## Build Dataset For Key Frame Selector

To build the dataset for Key Frame Selector, it is more convenient to build a npy file frist.
- for example, if you want to build a dataset using SSV2 with the checkpoint of 1600 epoch, you can run the following command
  ```bash
  python3 run_dataset.py \
      --img_path   'YOUR_PATH/SSV2/'  \
      --model_path 'YOUR_PATH/SAVE_DIR/checkpoint-1600.pth' \
      --fine_tune False
  ```

- After preparing the dataset, you should train the model 
  ```bash
  OUTPUT_DIR='/home/srtp_ghw/fqq_temp/output_dir'
  VIDEO_PATH='/home/srtp_ghw/fqq/data2/'
  MODEL_PATH="/home/srtp_ghw/fqq/MyMAE8/output_dir/checkpoint-1600.pth"
  python3 train_selector.py \
      --model pretrain_videomae_base_patch16_224 \
      --fine_tune True \
      --label "/home/srtp_ghw/fqq/50000.txt" \
      --data  "/home/srtp_ghw/fqq/sth_for_174.npy"
      ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
  
  ```

After that, we can evaluate the model by running the following commands.
- If you want to evaluate the model, you can run
  ```bash
  python3 test_selector.py \
      --selector_path 'YOUR_PATH/SAVE_DIR/checkpoint.pth'  \
      --model_path 'YOUR_PATH/SAVE_DIR/checkpoint-1600.pth' \
      --data_path 'YOUR_PATH/SSV2/'
  ```
- if you want to try maxpooling 49, then you should replace the codes in [dataset_build.py](./Frame Selector/dataset_build.py)
-       max_method = nn.MaxPool1d(kernel_size=49, stride=49)
        middle_layer = max_method(middle_layer)
        middle_layer = rearrange(middle_layer, 'b (c t) a -> b c t a', c=384, t=8)
- what's more, the model should also be modified simplily
- if you are going to use selector to compress a video and visulaze it, run the following
  ```bash
  python3 frame_selctor.py \
      

  ```
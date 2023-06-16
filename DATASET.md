# DATASET BUILD
## Build Dataset For Frame Selector

To build the dataset for frame selector, it is more convenient to build a npy file frist.
- for example, if you want to build a dataset using SSV2 with the checkpoint of 1600 epoch, you can run the following command
  ```bash
  python3 run_dataset.py \
      --img_path   'YOUR_PATH/SSV2/'  \
      --model_path 'YOUR_PATH/SAVE_DIR/checkpoint-1600.pth' \
      --fine_tune False
  ```
  
After that, we can evaluate the pretraining situation by running 
- nothing 
  ```bash
  python3 test_selector.py \
      --selector_path 'YOUR_PATH/SAVE_DIR/checkpoint.pth'  \
      --model_path 'YOUR_PATH/SAVE_DIR/checkpoint-1600.pth' \
      --data_path 'YOUR_PATH/SSV2/'
  ```
- if you are going to use selector to compress a video and visulaze it, run the following
  ```bash
  python3 frame_selctor.py \
      

  ```
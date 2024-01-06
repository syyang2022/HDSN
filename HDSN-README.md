# HDSN
一、项目名称与简介
Title:Pragmatic Degradation Learning for Scene Text Image Super-Resolution with data-training strategy

Our project is dedicated to the use of degradation processes and corresponding finetune strategies to enhance super-resolution effects.


二、文件夹列表
│  main.py
│  model_best.pth
│  requirement.txt
│  SDML-Readme.docx
│  
├─checkpoint
├─config
│      super_resolution.yaml
│      
├─dataset
│  │  create_lmdb.py
│  │  crop_800k.py
│  │  dataset.py
│  │  degra-dataset.py
│  │  stn_head.py
│  │  tps_spatial_transformer.py
│  │  voc_data.py
│  │  __init__.py
│  │  
│  └─mydata
│      │  confuse.pkl
│      │  crnn2.pth
│      │  demo.pth.tar
│      │  moran.pth
│      │  pretrain_transformer.pth
│      │  
│      ├─test
│      ├─train1
│      ├─train2
│      ├─train3
│      ├─train4  
│      ├─train5
│      └─train6
├─interfaces
│  │  base.py
│  │  degra-base.py
│  │  degra-super_resolution.py
│  │  stn_head.py
│  │  super_resolution-esrgan.py
│  │  super_resolution.py
│  │  tps_spatial_transformer.py  
├─loss
│  │  gradient_loss.py
│  │  percptual_loss.py
│  │  text_focus_loss.py
│  │  transformer.py
│  │  weight_ce_loss.py   
├─model
│  │  attention_recognition_head.py
│  │  bicubic.py
│  │  common.py
│  │  hdsn.py
│  │  stn_head.py
│  │  tools.py
│  │  tps_spatial_transformer.py
│  │  transformer2.py
│  │  
│  ├─aster
│  │      aster.pth.tar
│  │      
│  ├─crnn
│  │  │  crnn.py
│  │  │  __init__.py
│  ├─moran
│  │  │  asrn_res.py
│  │  │  fracPickup.py
│  │  │  moran.py
│  │  │  morn.py
│  │  │  __init__.py     
│  ├─recognizer
│  │  │  attention_recognition_head.py
│  │  │  recognizer_builder.py
│  │  │  resnet_aster.py
│  │  │  sequenceCrossEntropyLoss.py
│  │  │  stn_head.py
│  │  │  tps_spatial_transformer.py  
└─utils
    │  calculate_PSNR_SSIM.py
    │  labelmaps.py
    │  meters.py
    │  metrics.py
    │  ssim_psnr.py
    │  transformer2.py
    │  util.py
    │  utils_crnn.py
    │  utils_image.py
    │  utils_moran.py


三、所需环境
easydict==1.9
editdistance==0.5.3
lmdb==1.2.1
matplotlib==3.3.4
numpy
opencv-python==4.5.2.52
Pillow==8.2.0
six
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
torch==1.2.0
torchvision==0.2.1
tqdm==4.61.0
pyyaml
ipython
future
四、使用方法（★极其重要）
python main.py --batch_size=16 --exp_name=test --text_focus --test --resume='./checkpoint/yourfile/model_best.pth' 

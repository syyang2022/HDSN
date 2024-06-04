# 1、 Project Name and Introduction


Title:Pragmatic Degradation Learning for Scene Text Image Super-Resolution with data-training strategy

Paper link：https://doi.org/10.1016/j.knosys.2023.111349  

Abstract  
Super-resolution of scene text images represents a formidable computational problem, marred by a myriad of intricate challenges. This paper focuses on the specific hurdles that have impeded significant advancements in this domain, and introduces the Higher-Order Degradation-Based Super-Resolution Network (HDSN) as a novel solution to address these intricate issues. The challenges in super-resolving scene text images are manifold. Firstly, the semantic ambiguity inherent to text in natural scenes often leads to degraded results, as standard super-resolution techniques struggle to preserve meaningful textual content. Additionally, the uncertainty surrounding font variability exacerbates this issue, as different fonts require distinct treatment for optimal super-resolution. Furthermore, scene text images often exhibit long trailing shadows, artifacts, and strong noise, rendering conventional methods inadequate in producing satisfactory results. To tackle these intricate challenges, we propose a pragmatic higher-order degradation modeling process. This process takes into account the nuanced characteristics of scene text images, including the diverse forms of noise such as Gaussian, Poisson, speckle, and JPEG compression noise, as well as varying levels of blurring. By meticulously considering these real-world scenarios, our approach significantly enhances the robustness and adaptability of super-resolution for scene text images. In addition to addressing these challenges, we recognize the issues arising from sparse datasets and the lack of corresponding paired images for training. To surmount this limitation, we introduce a text image pre-training strategy, which proves to be highly effective in improving recognition accuracy. The experimental results on TextZoom affirm the effectiveness of our approach, demonstrating substantial improvements over existing methods. Notably, our HDSN achieves average recognition rates of 67.2% on ASTER, 63.2% on MORAN, and 58.0% on CRNN, surpassing the performance of available approaches.

# 2、 Folder List

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


# 3、 Required environment
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
# 4、 Operate (★important)
python main.py --batch_size=16 --exp_name=test --text_focus --test --resume='./checkpoint/yourfile/model_best.pth'  
# Citation
If our code or models help your work, please cite our paper:  
@article{YANG2024111349,  
title = {Pragmatic degradation learning for scene text image super-resolution with  
data-training strategy},  
journal = {Knowledge-Based Systems},  
volume = {285},  
pages = {111349},  
year = {2024},  
issn = {0950-7051},  
doi = {https://doi.org/10.1016/j.knosys.2023.111349},  
url = {https://www.sciencedirect.com/science/article/pii/S0950705123010973},  
author = {Shengying Yang and Lifeng Xie and Xiaoxiao Ran and Jingsheng Lei and Xiaohong  
Qian},

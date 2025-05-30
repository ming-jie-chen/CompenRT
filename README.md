# CompenRT
## Introduction
This is PyTorch implementation of the paper“Real-Time High-Resolution Projector Compensation”.

        
## Datasets
The high resolution compensation datasets:
[CompenHR](https://github.com/cyxwang/CompenHR/tree/main?tab=readme-ov-file#datasets) by [cyxwang]  

## Usage
   1. Clone this repo:
  
     git clone https://github.com/ming-jie-chen/CompenRT
     
   2. This project provides two versions of the input resolution: one 256 and one 512, which represent the model with the input of the photometric compensation subnetwork at 256×256 and 512×512 resolutions, 
      respectively, and you can choose one for training
      
     cd CompenRT/256 or cd CompenRT/512
     
     cd src/python  
     
   4. Download dataset and extract to ‘data/’
     
   5. Start visdom by typing:
      
     visdom

   5. Run train_CompenRT.py to produce results:
      
     python train_CompenRT.py

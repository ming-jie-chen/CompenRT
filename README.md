# CompenRT
## Introduction
This is PyTorch implementation of the paper“[(2025 ICME)Real-Time High-Resolution Projector Compensation]”.

        
## Datasets
The high resolution compensation datasets:
[CompenHR](https://github.com/cyxwang/CompenHR/tree/main?tab=readme-ov-file#datasets) by [cyxwang]  

## Usage
   1. Clone this repo:
  
     git clone https://github.com/ming-jie-chen/CompenRT
     
     cd CompenRT/256

   2. Download dataset and extract to ‘data/’
     
   3. Start visdom by typing:
      
     visdom

   4. Run train_CompenRT.py to produce results:
      
     python train_CompenRT.py

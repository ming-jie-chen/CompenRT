# CompenRT
## Introduction
This is PyTorch implementation of the paper“Real-Time High-Resolution Projector Compensation”.

        
## Datasets
The high resolution compensation datasets: One is the CompenHR public dataset and the other is the new high-resolution dataset we capture.

CompenHR public dataset:[CompenHR](https://github.com/cyxwang/CompenHR/tree/main?tab=readme-ov-file#datasets) by [cyxwang]  

New high-resolution dataset:
[cloud](https://drive.google.com/drive/folders/1ZRQgqKLQYdgaKeW8OsRLaNIbMlgyzZ-_?usp=drive_link),
[star](https://drive.google.com/drive/folders/17LYyJDYeQR8xsGAIzlXv5LKLFdro6J7r?usp=drive_link),
[yellow_paint](https://drive.google.com/drive/folders/1vwjYbXn92-7b8qdSpD8QL12Et3DR4vOo?usp=drive_link),

## Usage
   1. Clone this repo:
  
     git clone https://github.com/ming-jie-chen/CompenRT
     
   2. This project provides two versions of the input resolution: one 256 and one 512, which represent the model with the input of the photometric compensation subnetwork at 256×256 and 512×512 resolutions, 
      respectively, and you can choose one for training
      
     cd CompenRT/256 or cd CompenRT/512
     
     cd src/python  
     
   3. Download dataset and extract to ‘data/’
     
   4. Start visdom by typing:
      
     visdom

   5. Run train_CompenRT.py to produce results:
      
     python train_CompenRT.py
## Citation

        @inproceedings{chen2025compenrt,
           title      = {Real-Time High-Resolution Projector Compensation},
           booktitle  = {2025 IEEE International Conference on Multimedia & Expo (ICME)},
           author     = {Chen, Mingjie and Huang, bingyao},
           year       = {2025},
           publisher  = {IEEE},
           address    = {France, Nantes},
        }

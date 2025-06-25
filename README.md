# CompenRT
## Introduction
This is PyTorch implementation of the paper“Real-Time High-Resolution Projector Compensation”.

        
## Datasets
* [Full compensation](https://pan.baidu.com/s/1XblKFnsIBhjd2sK7aQCNDA?pwd=qnh5) (19 setups,same as [CompenHR](https://github.com/cyxwang/CompenHR) dataset)


## Usage
   1. Clone this repo:
  
     git clone https://github.com/ming-jie-chen/CompenRT
     
   2. This project provides two versions of the input resolution: CompenRT (256->1024) and CompenRT (512->1024), which represent the model with the input of the photometric compensation subnetwork at 256×256 and 512×512 resolutions, 
      respectively, and you can train:
      
     cd CompenRT/src/python  
     
   3. Download CompenRT [benchmark dataset](https://pan.baidu.com/s/1XblKFnsIBhjd2sK7aQCNDA?pwd=qnh5) and extract to [‘data/’](https://github.com/ming-jie-chen/CompenRT/tree/master/data)
     
   4. Start visdom by typing:
      
     visdom

   5. Open [`train_compenRT.py`](src/python/train_compenRT.py) and set which GPUs to use. An example is shown below, we use GPU 0.
   `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`

   6. Run [`train_compenRT.py`](src/python/train_compenRT.py) to reproduce benchmark results. To visualize the training process in **visdom** (slower), you need to set `plot_on=True`.
   
     python train_compenRT.py
   
## Citation
```
@inproceedings{chen2025compenrt,
        title      = {Real-Time High-Resolution Projector Compensation},
        booktitle  = {2025 IEEE International Conference on Multimedia & Expo (ICME)},
        author     = {Chen, Mingjie and Huang, Bingyao},
        year       = {2025},
        month      = {June},
        publisher  = {IEEE},
        address    = {France, Nantes},
}
```
## Acknowledgments
- This code borrows heavily from
  - The PyTorch implementation of CompenNeStPlusplusDataset.py, ImgProc.py, trainNetwork.py, utils.py is modified from [BingyaoHuang/CompenNeSt-plusplus](https://github.com/BingyaoHuang/CompenNeSt-plusplus/tree/master)
  - The PyTorch implementation of train_compenRT.py, test_compenRT, Models.py is modified from [cyxwang/CompenHR](https://github.com/cyxwang/CompenHR/tree/main)
  - The PyTorch implementation of TPS warping is modified from [cheind/py-thin-plate-spline](https://github.com/cheind/py-thin-plate-spline).
  - [ZhengyuZhao/differential color](https://github.com/ZhengyuZhao/PerC-Adversarial/blob/master/differential_color_functions.py) for differential_color_functions.py.
  - [Po-Hsun-Su/SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim) for for PyTorch implementation of SSIM loss.
  - [BingyaoHuang/CompenNeSt-plusplus](https://github.com/BingyaoHuang/CompenNeSt-plusplus) for data loader.
- We thank the authors of CompenHR for providing the code (count_hook.py, profile1.py) for statistical parameter quantities and FLOPs.
- We thank Jijiang Li for valuable discussion, proof reading, and help with the experiments.
- Feel free to open an issue if you have any questions/suggestions/concerns 😁.

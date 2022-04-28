# Ada-CM

This is the official PyTorch implementation of our CVPR 2022 paper:

> [**Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin**](https://arxiv.org/abs/2203.12341)      
> Hangyu Li, Nannan Wang*, Xi Yang, Xiaoyu Wang, Xinbo Gao        
> *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022*

## Update
- **(2022/04/28)** We upload the main files for training on RAF-DB.

## Requirements

We test the codes in the following environments, other versions may also be compatible:

- Python==3.6.9
- PyTorch==1.3.0
- Torchvision==0.4.1

## Dataset setup

Please download the dataset from [RAF-DB](http://www.whdeng.cn/raf/model1.html) website and change the root to your path. 

## Getting Started

To train on RAF-DB:

```bash
python main.py
```

## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{li2022adacm,
      title={Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin},
      author={Li, Hangyu and Wang, Nannan and Yang, Xi and Wang, Xiaoyu and Gao, Xinbo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }

## Note

If you have any questions, please contact me.  Email:  hangyuli.xidian@gmail.com.

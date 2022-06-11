# Ada-CM

This is the official PyTorch implementation of our CVPR 2022 paper:

> [**Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin**](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Towards_Semi-Supervised_Deep_Facial_Expression_Recognition_With_an_Adaptive_Confidence_CVPR_2022_paper.html)      
> Hangyu Li, Nannan Wang*, Xi Yang, Xiaoyu Wang, Xinbo Gao        
> *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022*

## Update
- **(2022/04/28)** We upload the main files for training on RAF-DB.
- **(2022/05/10)** We provide the pre-trained [ResNet-18](https://drive.google.com/file/d/17Jbg3G31uLTlwoB3thHlGrXD7BCH9CJZ/view?usp=sharing) on MS-Celeb-1M. Please put it into ***./models***.

## Requirements

We test the codes in the following environments, other versions may also be compatible:

- Python==3.6.9
- PyTorch==1.3.0
- Torchvision==0.4.1

## Dataset setup

Please download the dataset from [RAF-DB](http://www.whdeng.cn/raf/model1.html) website and change the root to your path. The splits of different-class labeled data are listed below. For example, for the case of 100 labels, the labeled training set consists of 10 faces annotated with fear and 15 faces annotated with other expressions. More details could be found in the supplementary material, which is available at CVF website. 

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=1>100 labels</th>
        <th align="center" colspan=1>400 labels</th>
        <th align="center" colspan=1>1000 labels</th>
        <th align="center" colspan=1>2000 labels</th>
        <th align="center" colspan=1>4000 labels</th>
    </tr>
    <tr>
        <td align="left">Fear</td>
        <td align="center">10</td>
        <td align="center">40</td>
        <td align="center">100</td>
        <td align="center">200</td>
        <td align="center">250</td>
    </tr>
    <tr>
        <td align="left">Others</td>
        <td align="center">15</td>
        <td align="center">60</td>
        <td align="center">150</td>
        <td align="center">300</td>
        <td align="center">625</td>
    </tr>
</table>

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
      month={June},
      year={2022},
      pages={4166-4175}
    }

## Note

If you have any questions, please contact me.  Email:  hangyuli.xidian@gmail.com

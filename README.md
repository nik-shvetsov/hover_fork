# This is branch for inference only

# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

This is an extended version of our previous work: XY-Net.  <br />

[Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) to Medical Image Analysis paper. 
[Link to paper](https://arxiv.org/abs/1812.06499v4)


## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}
```

## Overlaid Segmentation and Classification Prediction

<p float="left">
  <img src="/seg.gif" alt="Segmentation" width="870" />
</p>

The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)

## Companion Sites
The same version of this repository is officially available on the following sites for collection/affiliation purpose

* [Tissue Image Analytics Lab](https://github.com/TIA-Lab)
* [Quantitative Imaging and Informatics Laboratory](https://github.com/QuIIL)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
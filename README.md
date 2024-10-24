# TRIPS

This project hosts the code for implementing the TRIPS algorithm for the Domain-Generalized Class-Incremental Learning (DGCIL) task, as presented in our paper:

## [Multivariate Prototype Representation for Domain-Generalized Incremental Learning]

Can Peng, Piotr Koniusz, Kaiyu Guo, Brian C. Lovell, Peyman Moghadam.

[arXiv preprint](https://arxiv.org/pdf/2309.13563.pdf).

## Installation
This DGCIL_TRIPS implementation is based on DomainBed. Therefore the installation is the same as the original DomainBed benchmark.

Please check [requirements](https://github.com/csiro-robotics/DomainGeneralizedCIL/blob/main/requirements.txt) for environment requirements. 

You may also want to see the original README of [DomainBed](https://github.com/facebookresearch/DomainBed).

## Training

**train_all.py**: the training file for the base and incremental tasks. 

**run_script**: To conduct experiments using various datasets or methods, please refer to the respective shell script files located in the 'run_script' folder.

## Citations

Please consider citing the following paper in your publications if it helps your research.

```latexlatex
@article{peng2023multivariate,
  title={Multivariate Prototype Representation for Domain-Generalized Incremental Learning},
  author={Peng, Can and Koniusz, Piotr and Guo, Kaiyu and Lovell, Brian C and Moghadam, Peyman},
  journal={arXiv preprint arXiv:2309.13563},
  year={2023}
}
```

## Acknowledgements
Our project references the codes in the following repos. We thank the authors for making their code public.
* [DomainBed](https://github.com/facebookresearch/DomainBed)
* [SWAD](https://github.com/khanrc/swad)



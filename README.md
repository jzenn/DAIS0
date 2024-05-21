# Differentiable Annealed Importance Sampling Minimizes The Jensen-Shannon Divergence Between Initial and Target Distribution [(ICML 2024)](TODO)
<div id="top"></div>

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](TODO)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  <a href="https://jzenn.github.io" target="_blank">Johannes&nbsp;Zenn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a>


## About The Project
This is the official GitHub repository for our work [Differentiable Annealed Importance Sampling Minimizes The Jensen-Shannon Divergence Between Initial and Target Distribution](TODO) where we investigate the initial distribution of differentiable annealed importance sampling (DAIS) for inference.

> Differentiable annealed importance sampling (DAIS), proposed by Geffner & Domke (2021) and Zhang et al. (2023), allows optimizing, among others, over the initial distribution of AIS.
In this paper, we show that, in the limit of many transitions, DAIS minimizes the symmetrized KL divergence (Jensen-Shannon divergence) between the initial and target distribution.
Thus, DAIS can be seen as a form of variational inference (VI) in that its initial distribution is a parametric fit to an intractable target distribution. 
> In experiments on synthetic and real-world data, we observe that the initial distribution learned by DAIS often provides more accurate uncertainty estimates than standard VI (optimizing the reverse KL divergence), importance weighted VI, and Markovian score climbing (optimizing the forward KL divergence).

The code base was heavily extended and rewritten based on the DAIS code base by Zhang et al. (2021) which was taken from their [OpenReview submission](https://openreview.net/forum?id=6rqjgrL7Lq).
The baseline for Markovian Score Climbing (Naesseth et al., 2020) was taken from the [original repository](https://github.com/blei-lab/markovian-score-climbing/tree/main). 


## Environment: 

- tested with Python 3.10 (a lower version might work as well);
- dependencies can be found in `requirements.txt`


## Running the Code

Example training command for some logistic regression baseline:
```bash
python train_logistic_regression.py \
  --batch_size 128 \
  --n_particles 16 \
  --n_transitions 16 \
  --scaled_M \
  --max_iterations 10_000 \
  --lr 1e-3 \
  --dataset ionosphere \
  --data_path <path to dataset csv> \
  --path_to_save <path to save model to>
```

## License
Distributed under the MIT License. See `LICENSE.MIT` for more information.


## Citation:
Following is the Bibtex if you would like to cite our paper :

```bibtex
@inproceedings{zenn2024differentiable,
  title = {Differentiable Annealed Importance Sampling Minimizes The {J}ensen-{S}hannon Divergence Between Initial and Target Distribution},
  author = {Zenn, Johannes and Bamler, Robert},
  booktitle = {Forty-first International Conference on Machine Learning (ICML)},
  year = {2024},
  url = {https://openreview.net/pdf?id=rvaN2P1rvC}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
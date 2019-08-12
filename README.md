*Customized NAB for Multivariate data*

### Description
This project is based on the Numenta Anomaly Benchmark and provides a testing environment for multivariate streaming anomaly detection models.

Three models have been implemented:
- The subspace tracking model SPIRIT, by Papadimitriou et al. [link](http://www.cs.cmu.edu/afs/cs/project/spirit-1/www/)
- The kernel mean embedding model EXPoSE, by Schneider et al. [link](https://arxiv.org/abs/1601.06602)
- A custom Conditional kernel mean embedding Conditional EXPoSE, which was implemented for a master's thesis.

Next to some original NAB datasets, some synthetic datasets were generated.

### Quickstart guide:
1. Put your dataset in `data-preprocessed/`
2. Follow `srs/pipeline.py`
3. Results should be shown and stored in `results/`

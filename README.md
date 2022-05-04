# Long Sequence Time-Series Forecasting via Transformers

## File Structure

```shell
.
├── exp                 # Utils for serving data and running experiments
├── models              # Models and their building blocks
├── notebooks           # Jupyter notebooks for data generation and visualization
├── report              # LaTeX source for project final report
├── scripts             # Scripts that execute experiments
├── README.md
├── main.py             # Entry point of the program
└── requirements.yml    # Conda environment
```


## Usage

### Install dependencies

```shell
conda env create -f requirements.yml
conda activate lstf
```

### Ablation studies on benchmarks

The scripts for Transformers run vanilla Transformer (`enc-dec`), Transformer without the encoder (`dec`) and, Transformer with autoregressive decoding (`auto`). 
The scripts for Autoformer perform more detailed ablation studies. 

```shell
./scripts/run_transformer.sh
./scripts/run_autoformer.sh
```

### Ablation studies on synthetic dataset

```shell
./scripts/run_synthetic.sh
```


## References

+ [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
+ [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
+ [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)
+ [Transformers in Time Series: A Survey](https://arxiv.org/abs/2202.07125)

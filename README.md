# Long Sequence Time-Series Forecasting via Transformers

## File Structure

```shell
.
├── exp                 # Utils for serving data and running experiments
├── models              # Models and their building blocks
├── scripts             # Scripts and jupyter notebooks
├── README.md
├── main.py             # Entry point of the program
└── requirements.yml    # Conda environment
```


## Usage

### Install dependencies

```shell
conda env create -f requirements.yml
```

### Ablation studies on benchmarks

```shell
./scripts/run.sh
```

### Ablation studies on synthetic dataset

```shell
./scripts/run_synth.sh
```


## References

+ [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
+ [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)
+ [Transformers in Time Series: A Survey](https://arxiv.org/abs/2202.07125)

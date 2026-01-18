
<div align="center">
  <img src="assets/logo.png" alt="Logo">
</div>

**Markovian Pre-trained Transformer for Next-Item Recommendation:** ✅ 100% pre-trained on synthetic Markov chains; ✅ better transferability.

<div align="center">
  <img src="assets/vs.png" alt="VS">
</div>


## ⚙️ Requirements


```
conda create -n MPT python=3.10;conda activate MPT;bash setup.sh
```

## 🚀 Usage


```
┌── data # the 'root' path of data
│	├── Processed
│	│	├── Amazon2014Beauty_550_LOU # the training data
│	│	└── ...
│	├── Amazon2014Beauty.zip # the raw data
│	└── ...
|
├── logs # training logs
|
├── models # saving pre-trained models: e.g., sentence-t5-xl
|
├── configs
│	├── finetune.yaml # config for fine-tuning
│	└── pretrain.yaml # config for pre-training
|
├── encode.py # encoding item features
|
├── finetune.py
├── pretrain.py
|
└── sampler.py # sampling Markov trajectories
```


<div align="center">
  <img src="assets/overview.png" alt="overview">
</div>

### Markovian Pre-Training

    python pretrain.py --config configs/pretrain.yaml --alpha 0.05 --num-states 30


> [!TIP]
> The pre-trained models are stored in the `logs/...` directory.

### Recommendation Fine-Tuning

- **Adaptor:**

```
    python finetune.py --config configs/finetune.yaml --dataset Amazon2014Beauty_550_LOU --path logs/...
```

- **+LoRA:**

```
    python finetune.py --config configs/finetune.yaml --adaptor-only False --dataset Amazon2014Beauty_550_LOU --path logs/...
```

> [!NOTE]
> To reproduce the results presented in the paper, one should follow the steps outlined in [data/README.md](data/README.md) and [models/README.md](models/README.md) to download the processed datasets and pre-trained models.
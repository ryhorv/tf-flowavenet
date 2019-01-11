# FloWaveNet : A Generative Flow for Raw Audio

Unofficial tensorflow implementation of the paper ["FloWaveNet : A Generative Flow for Raw Audio".](https://arxiv.org/abs/1811.02155)

<img src="png/model.png">

## Requirements

- Python 3.5
- tensorflow 1.12
- Librosa

## How to use

1. Download the [LJ-Speech dataset](https://keithito.com/LJ-Speech-Dataset/) and unzip in to a new folder in the current dir.

2. Preprocess dataset using the following command: 

```python3 preprocessing.py --in_dir=dataset --out_dir=training_data```

where `dataset` is a folder with LJ-Speech dataset.

3. Run training: `python3 train.py`.


## Todo list

- [x] Multi-gpu training
- [x] Correct ActNorm initialization
- [x] Synthesis mode
- [ ] Global condition features


## Reference

 - Official pytorch implementation: [https://github.com/ksw0306/FloWaveNet](https://github.com/ksw0306/FloWaveNet)

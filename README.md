# FloWaveNet : A Generative Flow for Raw Audio

Unofficial tensorflow implementation of the paper ["FloWaveNet : A Generative Flow for Raw Audio".](https://arxiv.org/abs/1811.02155)

<img src="png/model.png">

## Requirements

- Python 3.5
- tensorflow 1.12
- Librosa

## How to use

1. Download the [LJ-Speech dataset](https://keithito.com/LJ-Speech-Dataset/) and unpack it:

```
>>> tar -xvf LJSpeech-1.1.tar.bz2
```

2. Preprocess dataset using the following command: 

```
>>> python3 preprocessing.py --in_dir=LJSpeech-1.1 --out_dir=training_data
```

3. Run training: 
```
>>> python3 train.py
```

## Features

- Implemented Multig-gpu training
- Added Global condition features
- Mixed precision training

With mixed precision training (enabled by default) the model can be trained for 7.5 days on a single GPU with 11Gb RAM. To use float32 training set `dtype=tf.float32` and `scale=1.` in `hparams.py`.

Several examples of synthesis can be found [here](examples).

## Todo list

- [ ] Learning rate and batch size tuning for efficient multi-GPU training


## Reference

 - Official pytorch implementation: [https://github.com/ksw0306/FloWaveNet](https://github.com/ksw0306/FloWaveNet)

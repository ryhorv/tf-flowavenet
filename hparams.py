import tensorflow as tf 
import numpy as np 


# Default hyperparameters
hparams = tf.contrib.training.HParams(
    num_gpus = 1, #Determines the number of gpus in use
    ps_device_type = 'GPU', # 'CPU'/'GPU'  Where gradients will sync

    #Audio
    num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
    rescaling_max = 0.999, #Rescaling value

    #Mel spectrogram
    n_fft = 1024, #Extra window size is filled with 0 paddings to match this parameter
    hop_size = 256, #For 22050Hz, 275 ~= 12.5 ms
    sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset)
    
    #Limits
    min_level_db = -100,
    ref_level_db = 20,
    fmin = 75, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
    fmax = 7600,
    
    max_time_steps = 6400,
    
    eval_max_time_steps = 22050 * 4,
    eval_samples = 1,

    split_random_state = 123,
    shuffle_random_seed = 42,
    test_size = 10,
    batch_size = 8,

    gin_channels = -1,
    n_speakers = 7,

    causal = False,
    n_block = 8,
    n_flow = 6,
    n_layer = 2,
    affine = True,
    causality = False,
    tf_random_seed = 75,
    temp = 0.7,
    upsample_scales  = [16, 16]
    )
import tensorflow as tf 
import numpy as np 


# Default hyperparameters
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners".
    cleaners='basic_cleaners',

    #Hardware setup
    num_gpus = 2, #Determines the number of gpus in use
    ps_device_type = 'GPU', # 'CPU'/'GPU'  Where gradients will sync
    use_gready_placement_startegy = False,
    use_nccl = False,
    moving_agerage_decay = 0.9999,
    ###########################################################################################################################################

    #Audio
    num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq = 513, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    rescaling_max = 0.999, #Rescaling value

    #Mel spectrogram
    n_fft = 512, #Extra window size is filled with 0 paddings to match this parameter
    hop_size = 96, #For 22050Hz, 275 ~= 12.5 ms
    win_size = 512, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
    sample_rate = 8000, #22050 Hz (corresponding to ljspeech dataset)
    frame_shift_ms = None,
    
    #train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
    clip_mels_length = True, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
    max_mel_frames = 1250,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

    #Limits
    min_level_db = -100,
    ref_level_db = 20,
    max_abs_value = 4.,
    fmin = 125, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
    fmax = 4000, 

    #Griffin Lim
    power = 1.5, 
    griffin_lim_iters = 60,
    ###########################################################################################################################################

    #Tacotron
    outputs_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
    stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them

    gin_channels = -1,
    n_speakers = 7,

    embedding_dim = 512, #dimension of embedding space
    enc_conv_num_layers = 3, #number of encoder convolutional layers
    enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
    enc_conv_channels = 512, #number of encoder convolutions filters for each layer
    encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

    smoothing = False, #Whether to smooth the attention normalization function 
    attention_dim = 128, #dimension of attention space
    attention_filters = 32, #number of attention convolution filters
    attention_kernel = (31, ), #kernel size of attention convolution
    cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

    prenet_layers = [256, 256], #number of layers and number of units of prenet
    decoder_layers = 2, #number of decoder lstm layers
    decoder_lstm_units = 1024, #number of decoder lstm units on each layer
    max_iters = 2500, #Max decoder steps during inference (Just for safety from infinite loop cases)

    postnet_num_layers = 5, #number of postnet convolutional layers
    postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
    postnet_channels = 512, #number of postnet convolution filters for each layer

    mask_encoder = False, #whether to mask encoder padding while computing attention
    mask_decoder = False, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)

    cross_entropy_pos_weight = 20, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
    ###########################################################################################################################################

    #Tacotron Training
    tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
    tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)
    
    #Usually your GPU can handle 16x tacotron_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
    tacotron_synthesis_batch_size = 32, #This ensures GTA synthesis goes up to 40x faster than one sample at a time and uses 100% of your GPU computation power.

    tacotron_batch_size = 32, #number of training samples on each training steps
    tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
    tacotron_scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

    tacotron_test_batches = 10, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
    tacotron_data_random_state=1234, #random state for train test split repeatability
    
    tacotron_gradient_clip = 5,

    tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay = 50000, #Step at which learning decay starts
    tacotron_decay_steps = 20000, #Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate = 0.2, #learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate = 1e-3, #starting learning rate
    tacotron_final_learning_rate = 1e-5, #minimal learning rate

    tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
    tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
    tacotron_adam_epsilon = 1e-6, #AdamOptimizer beta3 parameter

    tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
    tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet

    natural_eval = False, #Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

    #Decoder RNN learning can take be done in one of two ways:
    #    Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
    #    Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
    #The second approach is inspired by:
    #Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    #Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    tacotron_teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    tacotron_teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_final_ratio = 0., #final teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_start_decay = 10000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_steps = 280000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_alpha = 0., #teacher forcing ratio decay rate. Relevant if mode='scheduled'
    ###########################################################################################################################################
    
    
    causal = False,
    n_block = 5,
    n_flow = 6,
    n_layer = 2,
    affine = True,
    causality = False,
    tf_random_seed = 75,
    temp = 0.7,
    upsample_scales  = [8, 12],

    #Eval sentences (if no eval file was specified, these sentences are used for eval)
    sentences = [
    # From July 8, 2017 New York Times:
    "Здравствуйте! Вы позвонили в пиццу \"Счастье\"!",
    "У вас очень шумно! Не могли бы вы продолжить диалог в более тихом месте?",
    "Что-нибудь ещё хотите заказать?",
    "Вынуждены вас известить, что мы временно не принимаем банковские карты к оплате!",
    "Минимальная сумма заказа для осуществления бесплатной доставки составляет триста рублей.",
    "Всего доброго! Звоните нам еще!",
    "У нас все блюда вкусные!",
    "Как я поняла вы хотите пиццу. С чем бы вы хотели пиццу: с мясом, морепродуктами, сыром или колбасками?",
    "На данный момент действует акция на пиццу-пирог закрытую пиццу \"цыпленок цыпа\". Стоимость со скидкой составляет двести тринадцать рублей. Добавить в заказ?",
    "Отлично! Триста семьдесят больших пицц мюнхенских на тонком тесте добавлены в корзину!",
    "Прыжок с переподвыподвертом",
    "Параллелограмм, являющийся гранью параллелепипеда, вписанного в сферу, всегда жаждал стать тетраэдром. Но видно, не судьба...",
    "У лукоморья дуб зелёный; Златая цепь на дубе том: И днём и ночью кот учёный  Всё ходит по цепи кругом; Идёт направо - песнь заводит, Налево - сказку говорит. Там чудеса: там леший бродит, Русалка на ветвях сидит;",
    "Скажи-ка, дядя, ведь не даром Москва, спаленная пожаром, Французу отдана?",
    "Тетрагидропиранилциклопентилтетрагидропиридопиридиновые вещества вызывают гиппопотомомонстросесквиппедалиофобию, по причине того, что ни один зряченюхослышащий человек не способен такое выговорить.",
    "Максим Сергеевич, добрый день! Мы готовы подать заявку на конкурс!",
    "Илья, добрый день! Я бы хотела обсудить с вами вопрос закрытия актов по договору на синтез речи."
    
    ]

    )

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)
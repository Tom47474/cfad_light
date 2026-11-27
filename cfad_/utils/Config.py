class Config:
    def __init__(self, root_path, data_path, data, features, freq, seq_len, pred_len,
                 model, checkpoints, d_model, d_core, d_ff, e_layers, learning_rate,
                 lradj, train_epochs, patience, batch_size, dropout, activation, use_norm,
                 num_workers, use_gpu, gpu, save_model, num_inputs, num_channels,
                 kernel_size, lstm_hidden_size, lstm_num_layers, gru_hidden_size, gru_num_layers,
                 mlp_hidden_size, mlp_output_size, mlp_num_layers, task_name='long_term_forecast'):
        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.features = features
        self.freq = freq
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = model
        self.checkpoints = checkpoints
        self.d_model = d_model
        self.d_core = d_core
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.learning_rate = learning_rate
        self.lradj = lradj
        self.train_epochs = train_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation
        self.use_norm = use_norm
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.save_model = save_model
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_output_size = mlp_output_size
        self.mlp_num_layers = mlp_num_layers
        self.scaler = None  # 用于保存MinMaxScaler实例
        self.task_name = task_name  # 添加task_name属性
        # 添加TimesNet需要的其他属性
        self.enc_in = 1
        self.c_out = 1
        self.top_k = 5
        self.num_kernels = 6
        self.label_len = seq_len // 2
        self.embed = 'timeF'


configs = Config(
    root_path='../dataset/rate_dataset/',
    data_path='ReserveSendRates_2minutes.csv',
    data='ReserveSendRates_2minutes',
    features='M',
    freq='h',
    seq_len=24,
    pred_len=1,
    model='light_model',
    checkpoints='./checkpoints/',
    d_model=128,
    d_core=96,
    d_ff=256,
    e_layers=3, # 原来是3
    learning_rate=0.001,
    lradj='cosine',
    train_epochs=100,
    patience=3,
    batch_size=32,
    dropout=0,
    activation='gelu',
    use_norm=True,
    num_workers=0,
    use_gpu=True,
    gpu='0',
    save_model=True,
    num_inputs=1, # TCN
    num_channels=[32, 8, 1],  # 32 8 1
    kernel_size=2,
    lstm_hidden_size=128, # LSTM  128
    lstm_num_layers=1,
    gru_hidden_size = 128, # GRU  64
    gru_num_layers = 3,
    mlp_hidden_size=64, # MLP
    mlp_output_size=1,
    mlp_num_layers=3

)
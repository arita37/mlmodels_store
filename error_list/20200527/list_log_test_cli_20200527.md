## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 665](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L665)<br />665..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 2, [Traceback at line 1845](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1845)<br />1845..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 3, [Traceback at line 1880](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1880)<br />1880..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 4, [Traceback at line 1920](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1920)<br />1920..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 5, [Traceback at line 1955](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1955)<br />1955..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 6, [Traceback at line 2000](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2000)<br />2000..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 7, [Traceback at line 2035](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2035)<br />2035..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 8, [Traceback at line 2092](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2092)<br />2092..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 9, [Traceback at line 2096](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2096)<br />2096..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
<br />    mpars['encoder'] = MLPEncoder()   #bug in seq2seq
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 424, in init_wrapper
<br />    model = PydanticModel(**{**nmargs, **kwargs})
<br />  File "pydantic/main.py", line 283, in pydantic.main.BaseModel.__init__
<br />pydantic.error_wrappers.ValidationError: 1 validation error for MLPEncoderModel
<br />layer_sizes
<br />  field required (type=value_error.missing)
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
<br />
<br />  dataset/json/benchmark.json 
<br />
<br />  Custom benchmark 
<br />
<br />  ['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'] 
<br />
<br />  json_path https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test01/ 
<br />
<br />  Model List [{'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}] 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/armdn/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />Using TensorFlow backend.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />If using Keras pass *_constraint arguments to layers.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_probability/python/distributions/mixture.py:154: Categorical.event_size (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-05-19.
<br />Instructions for updating:
<br />The `event_size` property is deprecated.  Use `num_categories` instead.  They have the same value, but `event_size` is misnamed.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Use tf.where in 2.0, which has the same broadcast rule as np.where
<br />Model: "sequential_1"
<br />_________________________________________________________________
<br />Layer (type)                 Output Shape              Param #   
<br />=================================================================
<br />LSTM_1 (LSTM)                (None, 60, 300)           362400    
<br />_________________________________________________________________
<br />LSTM_2 (LSTM)                (None, 60, 200)           400800    
<br />_________________________________________________________________
<br />LSTM_3 (LSTM)                (None, 60, 24)            21600     
<br />_________________________________________________________________
<br />LSTM_4 (LSTM)                (None, 12)                1776      
<br />_________________________________________________________________
<br />dense_1 (Dense)              (None, 10)                130       
<br />_________________________________________________________________
<br />mdn_1 (MDN)                  (None, 363)               3993      
<br />=================================================================
<br />Total params: 790,699
<br />Trainable params: 790,699
<br />Non-trainable params: 0
<br />_________________________________________________________________
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1f4e693a20> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 351837.7812
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 112ms/step - loss: 243866.7031
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 110ms/step - loss: 125703.8594
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 115ms/step - loss: 56830.5000
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 102ms/step - loss: 26820.0156
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 113ms/step - loss: 14468.3750
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 111ms/step - loss: 8931.7959
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 119ms/step - loss: 6108.8774
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 112ms/step - loss: 4533.9302
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 112ms/step - loss: 3590.5972
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[ 0.9402076  11.364317   11.1987     10.6459675  11.16087    13.180277
<br />  12.496542   12.762818   12.696332   12.644145   10.014399   13.33057
<br />  11.379092   14.10261    13.602318   12.444684   11.411893   12.728938
<br />  10.691897   11.623378   10.836381   11.958544   11.032677   15.568011
<br />  12.389126   13.840481    8.670763   11.315521   11.467555   12.180652
<br />  13.099701   12.875865   12.513611   12.127208   11.366718   12.795475
<br />  12.177515   13.585345   12.079737   13.354988   12.176366   14.401572
<br />  12.628235   13.757295   12.600054   11.513715   12.494611    9.977074
<br />  12.64993    11.271819   10.386806   12.077227    9.851064   12.089888
<br />  12.183039   11.403437   11.676581    8.629911   13.205508   12.824393
<br />   0.05464089 -0.4011411   0.27497554  0.4653784  -0.03300336  1.3449221
<br />  -0.69141984  1.6139327  -1.4326293  -0.7061124  -0.413397    1.0986676
<br />   0.19012782 -0.3729353   0.9111439  -1.9476631  -0.11212027  0.73379385
<br />   0.5492965  -0.97594684  0.29346126  0.42518243  0.725695   -0.41716123
<br />  -0.11342096  0.0180338   0.3789252  -2.1063888   1.5156311   0.8346741
<br />   0.81507117 -2.5647006   0.9315667  -0.13636413 -1.6197813   0.0859251
<br />   0.6110602   1.7330158  -1.8953689   0.2083194   0.03667194  0.30991513
<br />  -1.3690679   0.5470052  -0.68559265  1.1489235   0.07566464 -0.7426022
<br />  -1.309137   -0.6242819  -0.9750496   1.4055986   1.395261    0.8931333
<br />   0.29769295 -0.55322     2.018494    2.8773925  -1.5877116   2.34028
<br />   0.16035128  0.78026307 -0.51542586  0.29830515 -0.03585251  0.06955826
<br />   1.3616935  -0.59023124  1.466565    0.32088137  0.34916145  1.0216672
<br />  -0.2769941  -0.58793163 -0.09363353 -0.98131055 -0.89187133  0.8263006
<br />   1.863868   -0.05723858 -0.23842183  0.6916253  -0.2982387   2.1172633
<br />  -0.07185566  0.43379992 -0.9283353  -0.21514773  0.3241016   0.8094094
<br />  -1.7564788  -1.4199502   1.7094935  -1.5330927   1.2310821  -0.22579575
<br />   0.10351712  2.0236838  -0.19643037  1.6596265   1.7128298   1.1650527
<br />   0.7173665   1.2050571  -1.765499    0.55680025  0.5053262   0.05584989
<br />   0.22331208  0.65918165 -0.353078    0.63295084  0.51825917 -0.62670887
<br />   1.5309649   1.0286694   1.1164732  -1.0751774   0.6778823  -1.8793364
<br />   0.77101713 10.645291   10.281864   10.117427   13.405476   12.478419
<br />  13.472109   10.860724   13.300526   12.142553   12.502805   12.773365
<br />  11.94539    11.945209   14.903318    9.3841715  11.610393   11.386513
<br />  11.166038   10.76138    11.788466   13.301577   12.55526     9.907596
<br />  11.916902   12.368995   10.45019    10.129285   12.005425    8.894257
<br />  12.953056   13.049019   12.393717   11.34584    11.204836   10.809234
<br />  11.967758   11.905863   10.607189   10.713747   10.568208   12.708354
<br />  11.595296   12.135948   11.634887    9.804965   12.914345   11.766924
<br />  12.041777   13.437128   12.077129   11.017586   12.308876   10.88357
<br />  10.684696   13.053781   11.953948   11.234972   12.30826    11.621726
<br />   2.2738795   0.49702495  0.73969394  0.24809593  2.1453187   0.9559495
<br />   1.1998537   2.0067692   0.9725055   0.02439839  1.976815    0.22056538
<br />   0.13594323  1.2543637   0.81876254  0.7886519   1.6902626   1.2341866
<br />   1.8378258   2.3972597   0.7033515   0.34802568  3.136157    1.1988055
<br />   0.63778734  2.2882307   1.4997472   0.15013254  1.9660578   1.472682
<br />   2.1780043   1.3591049   1.6260297   1.4338082   1.7350852   0.5051892
<br />   1.2497014   2.6620288   0.10892963  0.91671824  1.1685748   1.0254155
<br />   0.0990569   1.6660585   0.5240454   0.86250514  0.5276132   0.7045972
<br />   0.13761193  0.89357525  0.26749712  0.1598537   0.8720962   0.18681502
<br />   1.9116867   0.18562144  0.41283178  0.59457964  1.1338948   1.2520875
<br />   1.1116236   2.5682373   1.8975787   3.0770273   1.9695115   1.5224731
<br />   1.4332272   0.9200427   0.23207533  0.417049    0.9877678   0.3848489
<br />   3.5340219   0.34857535  0.52038586  2.9255626   0.71989954  0.64407873
<br />   0.73010874  0.66725063  1.3690689   0.6231352   0.5262172   1.1573606
<br />   2.4295516   1.3498938   2.1069224   0.2100991   0.41557753  0.89301777
<br />   1.3862282   2.1571069   1.1372676   3.0596905   0.5399623   0.3367744
<br />   0.72948647  0.38773185  1.5519613   2.5120687   0.69330883  0.29940248
<br />   0.27809215  2.1460428   1.5470383   1.0475013   0.1613692   1.4260583
<br />   1.0779011   1.5260218   0.7280821   1.0544176   3.0075302   1.4714316
<br />   1.0964603   0.12271261  1.5872588   0.34134293  3.0438023   0.84644985
<br />  10.08839    -8.658146   -6.582844  ]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-26 23:57:26.454791
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   90.7434
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:57:26.459509
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                    8264.3
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:57:26.463913
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   90.3342
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:57:26.467813
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -739.127
<br />metric_name                                             r2_score
<br />Name: 3, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139771855782072
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139769641569472
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139769641569976
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139769641165040
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139769641165544
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139769641166048
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1f629db438> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.639943
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.600618
<br />grad_step = 000002, loss = 0.571953
<br />grad_step = 000003, loss = 0.541606
<br />grad_step = 000004, loss = 0.508621
<br />grad_step = 000005, loss = 0.474461
<br />grad_step = 000006, loss = 0.449516
<br />grad_step = 000007, loss = 0.442397
<br />grad_step = 000008, loss = 0.425279
<br />grad_step = 000009, loss = 0.403349
<br />grad_step = 000010, loss = 0.385713
<br />grad_step = 000011, loss = 0.370097
<br />grad_step = 000012, loss = 0.357842
<br />grad_step = 000013, loss = 0.347541
<br />grad_step = 000014, loss = 0.335305
<br />grad_step = 000015, loss = 0.320491
<br />grad_step = 000016, loss = 0.305254
<br />grad_step = 000017, loss = 0.291742
<br />grad_step = 000018, loss = 0.280452
<br />grad_step = 000019, loss = 0.269655
<br />grad_step = 000020, loss = 0.257701
<br />grad_step = 000021, loss = 0.245202
<br />grad_step = 000022, loss = 0.233414
<br />grad_step = 000023, loss = 0.222713
<br />grad_step = 000024, loss = 0.212571
<br />grad_step = 000025, loss = 0.202204
<br />grad_step = 000026, loss = 0.191450
<br />grad_step = 000027, loss = 0.180992
<br />grad_step = 000028, loss = 0.171323
<br />grad_step = 000029, loss = 0.162330
<br />grad_step = 000030, loss = 0.153724
<br />grad_step = 000031, loss = 0.145247
<br />grad_step = 000032, loss = 0.136886
<br />grad_step = 000033, loss = 0.128779
<br />grad_step = 000034, loss = 0.120892
<br />grad_step = 000035, loss = 0.113179
<br />grad_step = 000036, loss = 0.105787
<br />grad_step = 000037, loss = 0.098964
<br />grad_step = 000038, loss = 0.092661
<br />grad_step = 000039, loss = 0.086408
<br />grad_step = 000040, loss = 0.080113
<br />grad_step = 000041, loss = 0.074253
<br />grad_step = 000042, loss = 0.068966
<br />grad_step = 000043, loss = 0.063920
<br />grad_step = 000044, loss = 0.058982
<br />grad_step = 000045, loss = 0.054254
<br />grad_step = 000046, loss = 0.049855
<br />grad_step = 000047, loss = 0.045848
<br />grad_step = 000048, loss = 0.042050
<br />grad_step = 000049, loss = 0.038460
<br />grad_step = 000050, loss = 0.035124
<br />grad_step = 000051, loss = 0.031973
<br />grad_step = 000052, loss = 0.029123
<br />grad_step = 000053, loss = 0.026578
<br />grad_step = 000054, loss = 0.024202
<br />grad_step = 000055, loss = 0.021989
<br />grad_step = 000056, loss = 0.019973
<br />grad_step = 000057, loss = 0.018184
<br />grad_step = 000058, loss = 0.016534
<br />grad_step = 000059, loss = 0.015044
<br />grad_step = 000060, loss = 0.013704
<br />grad_step = 000061, loss = 0.012490
<br />grad_step = 000062, loss = 0.011407
<br />grad_step = 000063, loss = 0.010422
<br />grad_step = 000064, loss = 0.009565
<br />grad_step = 000065, loss = 0.008779
<br />grad_step = 000066, loss = 0.008053
<br />grad_step = 000067, loss = 0.007418
<br />grad_step = 000068, loss = 0.006866
<br />grad_step = 000069, loss = 0.006336
<br />grad_step = 000070, loss = 0.005861
<br />grad_step = 000071, loss = 0.005443
<br />grad_step = 000072, loss = 0.005069
<br />grad_step = 000073, loss = 0.004728
<br />grad_step = 000074, loss = 0.004421
<br />grad_step = 000075, loss = 0.004144
<br />grad_step = 000076, loss = 0.003897
<br />grad_step = 000077, loss = 0.003677
<br />grad_step = 000078, loss = 0.003479
<br />grad_step = 000079, loss = 0.003301
<br />grad_step = 000080, loss = 0.003152
<br />grad_step = 000081, loss = 0.003017
<br />grad_step = 000082, loss = 0.002899
<br />grad_step = 000083, loss = 0.002800
<br />grad_step = 000084, loss = 0.002711
<br />grad_step = 000085, loss = 0.002635
<br />grad_step = 000086, loss = 0.002574
<br />grad_step = 000087, loss = 0.002521
<br />grad_step = 000088, loss = 0.002472
<br />grad_step = 000089, loss = 0.002433
<br />grad_step = 000090, loss = 0.002401
<br />grad_step = 000091, loss = 0.002373
<br />grad_step = 000092, loss = 0.002349
<br />grad_step = 000093, loss = 0.002325
<br />grad_step = 000094, loss = 0.002306
<br />grad_step = 000095, loss = 0.002289
<br />grad_step = 000096, loss = 0.002272
<br />grad_step = 000097, loss = 0.002256
<br />grad_step = 000098, loss = 0.002242
<br />grad_step = 000099, loss = 0.002228
<br />grad_step = 000100, loss = 0.002214
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002201
<br />grad_step = 000102, loss = 0.002188
<br />grad_step = 000103, loss = 0.002176
<br />grad_step = 000104, loss = 0.002165
<br />grad_step = 000105, loss = 0.002154
<br />grad_step = 000106, loss = 0.002144
<br />grad_step = 000107, loss = 0.002134
<br />grad_step = 000108, loss = 0.002125
<br />grad_step = 000109, loss = 0.002117
<br />grad_step = 000110, loss = 0.002109
<br />grad_step = 000111, loss = 0.002103
<br />grad_step = 000112, loss = 0.002096
<br />grad_step = 000113, loss = 0.002091
<br />grad_step = 000114, loss = 0.002086
<br />grad_step = 000115, loss = 0.002081
<br />grad_step = 000116, loss = 0.002076
<br />grad_step = 000117, loss = 0.002072
<br />grad_step = 000118, loss = 0.002069
<br />grad_step = 000119, loss = 0.002065
<br />grad_step = 000120, loss = 0.002062
<br />grad_step = 000121, loss = 0.002059
<br />grad_step = 000122, loss = 0.002056
<br />grad_step = 000123, loss = 0.002052
<br />grad_step = 000124, loss = 0.002050
<br />grad_step = 000125, loss = 0.002047
<br />grad_step = 000126, loss = 0.002044
<br />grad_step = 000127, loss = 0.002041
<br />grad_step = 000128, loss = 0.002038
<br />grad_step = 000129, loss = 0.002035
<br />grad_step = 000130, loss = 0.002032
<br />grad_step = 000131, loss = 0.002030
<br />grad_step = 000132, loss = 0.002029
<br />grad_step = 000133, loss = 0.002027
<br />grad_step = 000134, loss = 0.002023
<br />grad_step = 000135, loss = 0.002018
<br />grad_step = 000136, loss = 0.002016
<br />grad_step = 000137, loss = 0.002015
<br />grad_step = 000138, loss = 0.002012
<br />grad_step = 000139, loss = 0.002008
<br />grad_step = 000140, loss = 0.002005
<br />grad_step = 000141, loss = 0.002004
<br />grad_step = 000142, loss = 0.002002
<br />grad_step = 000143, loss = 0.001999
<br />grad_step = 000144, loss = 0.001995
<br />grad_step = 000145, loss = 0.001993
<br />grad_step = 000146, loss = 0.001991
<br />grad_step = 000147, loss = 0.001989
<br />grad_step = 000148, loss = 0.001987
<br />grad_step = 000149, loss = 0.001985
<br />grad_step = 000150, loss = 0.001982
<br />grad_step = 000151, loss = 0.001979
<br />grad_step = 000152, loss = 0.001976
<br />grad_step = 000153, loss = 0.001974
<br />grad_step = 000154, loss = 0.001971
<br />grad_step = 000155, loss = 0.001969
<br />grad_step = 000156, loss = 0.001967
<br />grad_step = 000157, loss = 0.001965
<br />grad_step = 000158, loss = 0.001965
<br />grad_step = 000159, loss = 0.001967
<br />grad_step = 000160, loss = 0.001976
<br />grad_step = 000161, loss = 0.001992
<br />grad_step = 000162, loss = 0.001998
<br />grad_step = 000163, loss = 0.001986
<br />grad_step = 000164, loss = 0.001961
<br />grad_step = 000165, loss = 0.001949
<br />grad_step = 000166, loss = 0.001958
<br />grad_step = 000167, loss = 0.001970
<br />grad_step = 000168, loss = 0.001971
<br />grad_step = 000169, loss = 0.001956
<br />grad_step = 000170, loss = 0.001942
<br />grad_step = 000171, loss = 0.001940
<br />grad_step = 000172, loss = 0.001947
<br />grad_step = 000173, loss = 0.001953
<br />grad_step = 000174, loss = 0.001950
<br />grad_step = 000175, loss = 0.001940
<br />grad_step = 000176, loss = 0.001931
<br />grad_step = 000177, loss = 0.001928
<br />grad_step = 000178, loss = 0.001930
<br />grad_step = 000179, loss = 0.001934
<br />grad_step = 000180, loss = 0.001936
<br />grad_step = 000181, loss = 0.001933
<br />grad_step = 000182, loss = 0.001927
<br />grad_step = 000183, loss = 0.001921
<br />grad_step = 000184, loss = 0.001916
<br />grad_step = 000185, loss = 0.001914
<br />grad_step = 000186, loss = 0.001914
<br />grad_step = 000187, loss = 0.001915
<br />grad_step = 000188, loss = 0.001916
<br />grad_step = 000189, loss = 0.001918
<br />grad_step = 000190, loss = 0.001919
<br />grad_step = 000191, loss = 0.001921
<br />grad_step = 000192, loss = 0.001921
<br />grad_step = 000193, loss = 0.001923
<br />grad_step = 000194, loss = 0.001923
<br />grad_step = 000195, loss = 0.001924
<br />grad_step = 000196, loss = 0.001921
<br />grad_step = 000197, loss = 0.001918
<br />grad_step = 000198, loss = 0.001912
<br />grad_step = 000199, loss = 0.001907
<br />grad_step = 000200, loss = 0.001901
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001897
<br />grad_step = 000202, loss = 0.001892
<br />grad_step = 000203, loss = 0.001888
<br />grad_step = 000204, loss = 0.001886
<br />grad_step = 000205, loss = 0.001883
<br />grad_step = 000206, loss = 0.001881
<br />grad_step = 000207, loss = 0.001879
<br />grad_step = 000208, loss = 0.001878
<br />grad_step = 000209, loss = 0.001876
<br />grad_step = 000210, loss = 0.001875
<br />grad_step = 000211, loss = 0.001873
<br />grad_step = 000212, loss = 0.001872
<br />grad_step = 000213, loss = 0.001870
<br />grad_step = 000214, loss = 0.001869
<br />grad_step = 000215, loss = 0.001868
<br />grad_step = 000216, loss = 0.001867
<br />grad_step = 000217, loss = 0.001868
<br />grad_step = 000218, loss = 0.001875
<br />grad_step = 000219, loss = 0.001893
<br />grad_step = 000220, loss = 0.001949
<br />grad_step = 000221, loss = 0.002047
<br />grad_step = 000222, loss = 0.002241
<br />grad_step = 000223, loss = 0.002245
<br />grad_step = 000224, loss = 0.002116
<br />grad_step = 000225, loss = 0.001892
<br />grad_step = 000226, loss = 0.001924
<br />grad_step = 000227, loss = 0.002087
<br />grad_step = 000228, loss = 0.002030
<br />grad_step = 000229, loss = 0.001875
<br />grad_step = 000230, loss = 0.001885
<br />grad_step = 000231, loss = 0.001997
<br />grad_step = 000232, loss = 0.002039
<br />grad_step = 000233, loss = 0.001909
<br />grad_step = 000234, loss = 0.001844
<br />grad_step = 000235, loss = 0.001910
<br />grad_step = 000236, loss = 0.001967
<br />grad_step = 000237, loss = 0.001925
<br />grad_step = 000238, loss = 0.001846
<br />grad_step = 000239, loss = 0.001853
<br />grad_step = 000240, loss = 0.001905
<br />grad_step = 000241, loss = 0.001900
<br />grad_step = 000242, loss = 0.001847
<br />grad_step = 000243, loss = 0.001834
<br />grad_step = 000244, loss = 0.001864
<br />grad_step = 000245, loss = 0.001879
<br />grad_step = 000246, loss = 0.001845
<br />grad_step = 000247, loss = 0.001825
<br />grad_step = 000248, loss = 0.001838
<br />grad_step = 000249, loss = 0.001855
<br />grad_step = 000250, loss = 0.001842
<br />grad_step = 000251, loss = 0.001821
<br />grad_step = 000252, loss = 0.001820
<br />grad_step = 000253, loss = 0.001834
<br />grad_step = 000254, loss = 0.001834
<br />grad_step = 000255, loss = 0.001820
<br />grad_step = 000256, loss = 0.001811
<br />grad_step = 000257, loss = 0.001815
<br />grad_step = 000258, loss = 0.001821
<br />grad_step = 000259, loss = 0.001818
<br />grad_step = 000260, loss = 0.001808
<br />grad_step = 000261, loss = 0.001803
<br />grad_step = 000262, loss = 0.001805
<br />grad_step = 000263, loss = 0.001808
<br />grad_step = 000264, loss = 0.001806
<br />grad_step = 000265, loss = 0.001800
<br />grad_step = 000266, loss = 0.001795
<br />grad_step = 000267, loss = 0.001795
<br />grad_step = 000268, loss = 0.001797
<br />grad_step = 000269, loss = 0.001796
<br />grad_step = 000270, loss = 0.001793
<br />grad_step = 000271, loss = 0.001789
<br />grad_step = 000272, loss = 0.001786
<br />grad_step = 000273, loss = 0.001785
<br />grad_step = 000274, loss = 0.001785
<br />grad_step = 000275, loss = 0.001785
<br />grad_step = 000276, loss = 0.001783
<br />grad_step = 000277, loss = 0.001781
<br />grad_step = 000278, loss = 0.001778
<br />grad_step = 000279, loss = 0.001775
<br />grad_step = 000280, loss = 0.001774
<br />grad_step = 000281, loss = 0.001772
<br />grad_step = 000282, loss = 0.001772
<br />grad_step = 000283, loss = 0.001771
<br />grad_step = 000284, loss = 0.001769
<br />grad_step = 000285, loss = 0.001767
<br />grad_step = 000286, loss = 0.001765
<br />grad_step = 000287, loss = 0.001763
<br />grad_step = 000288, loss = 0.001761
<br />grad_step = 000289, loss = 0.001759
<br />grad_step = 000290, loss = 0.001758
<br />grad_step = 000291, loss = 0.001756
<br />grad_step = 000292, loss = 0.001754
<br />grad_step = 000293, loss = 0.001753
<br />grad_step = 000294, loss = 0.001751
<br />grad_step = 000295, loss = 0.001750
<br />grad_step = 000296, loss = 0.001748
<br />grad_step = 000297, loss = 0.001747
<br />grad_step = 000298, loss = 0.001746
<br />grad_step = 000299, loss = 0.001745
<br />grad_step = 000300, loss = 0.001745
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001747
<br />grad_step = 000302, loss = 0.001752
<br />grad_step = 000303, loss = 0.001764
<br />grad_step = 000304, loss = 0.001791
<br />grad_step = 000305, loss = 0.001849
<br />grad_step = 000306, loss = 0.001956
<br />grad_step = 000307, loss = 0.002154
<br />grad_step = 000308, loss = 0.002358
<br />grad_step = 000309, loss = 0.002487
<br />grad_step = 000310, loss = 0.002221
<br />grad_step = 000311, loss = 0.001844
<br />grad_step = 000312, loss = 0.001739
<br />grad_step = 000313, loss = 0.001954
<br />grad_step = 000314, loss = 0.002112
<br />grad_step = 000315, loss = 0.001936
<br />grad_step = 000316, loss = 0.001731
<br />grad_step = 000317, loss = 0.001803
<br />grad_step = 000318, loss = 0.001954
<br />grad_step = 000319, loss = 0.001896
<br />grad_step = 000320, loss = 0.001736
<br />grad_step = 000321, loss = 0.001755
<br />grad_step = 000322, loss = 0.001869
<br />grad_step = 000323, loss = 0.001839
<br />grad_step = 000324, loss = 0.001726
<br />grad_step = 000325, loss = 0.001733
<br />grad_step = 000326, loss = 0.001806
<br />grad_step = 000327, loss = 0.001793
<br />grad_step = 000328, loss = 0.001717
<br />grad_step = 000329, loss = 0.001717
<br />grad_step = 000330, loss = 0.001766
<br />grad_step = 000331, loss = 0.001758
<br />grad_step = 000332, loss = 0.001708
<br />grad_step = 000333, loss = 0.001705
<br />grad_step = 000334, loss = 0.001737
<br />grad_step = 000335, loss = 0.001736
<br />grad_step = 000336, loss = 0.001702
<br />grad_step = 000337, loss = 0.001694
<br />grad_step = 000338, loss = 0.001715
<br />grad_step = 000339, loss = 0.001718
<br />grad_step = 000340, loss = 0.001697
<br />grad_step = 000341, loss = 0.001685
<br />grad_step = 000342, loss = 0.001695
<br />grad_step = 000343, loss = 0.001703
<br />grad_step = 000344, loss = 0.001693
<br />grad_step = 000345, loss = 0.001680
<br />grad_step = 000346, loss = 0.001680
<br />grad_step = 000347, loss = 0.001687
<br />grad_step = 000348, loss = 0.001687
<br />grad_step = 000349, loss = 0.001677
<br />grad_step = 000350, loss = 0.001671
<br />grad_step = 000351, loss = 0.001672
<br />grad_step = 000352, loss = 0.001676
<br />grad_step = 000353, loss = 0.001674
<br />grad_step = 000354, loss = 0.001667
<br />grad_step = 000355, loss = 0.001662
<br />grad_step = 000356, loss = 0.001662
<br />grad_step = 000357, loss = 0.001664
<br />grad_step = 000358, loss = 0.001663
<br />grad_step = 000359, loss = 0.001659
<br />grad_step = 000360, loss = 0.001655
<br />grad_step = 000361, loss = 0.001653
<br />grad_step = 000362, loss = 0.001653
<br />grad_step = 000363, loss = 0.001653
<br />grad_step = 000364, loss = 0.001651
<br />grad_step = 000365, loss = 0.001648
<br />grad_step = 000366, loss = 0.001645
<br />grad_step = 000367, loss = 0.001643
<br />grad_step = 000368, loss = 0.001641
<br />grad_step = 000369, loss = 0.001640
<br />grad_step = 000370, loss = 0.001640
<br />grad_step = 000371, loss = 0.001638
<br />grad_step = 000372, loss = 0.001636
<br />grad_step = 000373, loss = 0.001634
<br />grad_step = 000374, loss = 0.001632
<br />grad_step = 000375, loss = 0.001629
<br />grad_step = 000376, loss = 0.001627
<br />grad_step = 000377, loss = 0.001626
<br />grad_step = 000378, loss = 0.001624
<br />grad_step = 000379, loss = 0.001623
<br />grad_step = 000380, loss = 0.001621
<br />grad_step = 000381, loss = 0.001620
<br />grad_step = 000382, loss = 0.001619
<br />grad_step = 000383, loss = 0.001618
<br />grad_step = 000384, loss = 0.001618
<br />grad_step = 000385, loss = 0.001618
<br />grad_step = 000386, loss = 0.001620
<br />grad_step = 000387, loss = 0.001625
<br />grad_step = 000388, loss = 0.001635
<br />grad_step = 000389, loss = 0.001655
<br />grad_step = 000390, loss = 0.001695
<br />grad_step = 000391, loss = 0.001770
<br />grad_step = 000392, loss = 0.001899
<br />grad_step = 000393, loss = 0.002070
<br />grad_step = 000394, loss = 0.002202
<br />grad_step = 000395, loss = 0.002182
<br />grad_step = 000396, loss = 0.001926
<br />grad_step = 000397, loss = 0.001659
<br />grad_step = 000398, loss = 0.001619
<br />grad_step = 000399, loss = 0.001781
<br />grad_step = 000400, loss = 0.001900
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001818
<br />grad_step = 000402, loss = 0.001641
<br />grad_step = 000403, loss = 0.001599
<br />grad_step = 000404, loss = 0.001699
<br />grad_step = 000405, loss = 0.001781
<br />grad_step = 000406, loss = 0.001727
<br />grad_step = 000407, loss = 0.001617
<br />grad_step = 000408, loss = 0.001585
<br />grad_step = 000409, loss = 0.001639
<br />grad_step = 000410, loss = 0.001696
<br />grad_step = 000411, loss = 0.001673
<br />grad_step = 000412, loss = 0.001606
<br />grad_step = 000413, loss = 0.001574
<br />grad_step = 000414, loss = 0.001600
<br />grad_step = 000415, loss = 0.001637
<br />grad_step = 000416, loss = 0.001632
<br />grad_step = 000417, loss = 0.001596
<br />grad_step = 000418, loss = 0.001568
<br />grad_step = 000419, loss = 0.001573
<br />grad_step = 000420, loss = 0.001595
<br />grad_step = 000421, loss = 0.001603
<br />grad_step = 000422, loss = 0.001588
<br />grad_step = 000423, loss = 0.001566
<br />grad_step = 000424, loss = 0.001557
<br />grad_step = 000425, loss = 0.001564
<br />grad_step = 000426, loss = 0.001575
<br />grad_step = 000427, loss = 0.001576
<br />grad_step = 000428, loss = 0.001566
<br />grad_step = 000429, loss = 0.001554
<br />grad_step = 000430, loss = 0.001547
<br />grad_step = 000431, loss = 0.001549
<br />grad_step = 000432, loss = 0.001554
<br />grad_step = 000433, loss = 0.001557
<br />grad_step = 000434, loss = 0.001554
<br />grad_step = 000435, loss = 0.001547
<br />grad_step = 000436, loss = 0.001540
<br />grad_step = 000437, loss = 0.001535
<br />grad_step = 000438, loss = 0.001534
<br />grad_step = 000439, loss = 0.001535
<br />grad_step = 000440, loss = 0.001537
<br />grad_step = 000441, loss = 0.001537
<br />grad_step = 000442, loss = 0.001535
<br />grad_step = 000443, loss = 0.001531
<br />grad_step = 000444, loss = 0.001528
<br />grad_step = 000445, loss = 0.001524
<br />grad_step = 000446, loss = 0.001520
<br />grad_step = 000447, loss = 0.001518
<br />grad_step = 000448, loss = 0.001516
<br />grad_step = 000449, loss = 0.001514
<br />grad_step = 000450, loss = 0.001513
<br />grad_step = 000451, loss = 0.001512
<br />grad_step = 000452, loss = 0.001512
<br />grad_step = 000453, loss = 0.001512
<br />grad_step = 000454, loss = 0.001512
<br />grad_step = 000455, loss = 0.001514
<br />grad_step = 000456, loss = 0.001517
<br />grad_step = 000457, loss = 0.001525
<br />grad_step = 000458, loss = 0.001539
<br />grad_step = 000459, loss = 0.001567
<br />grad_step = 000460, loss = 0.001619
<br />grad_step = 000461, loss = 0.001714
<br />grad_step = 000462, loss = 0.001855
<br />grad_step = 000463, loss = 0.002050
<br />grad_step = 000464, loss = 0.002150
<br />grad_step = 000465, loss = 0.002103
<br />grad_step = 000466, loss = 0.001800
<br />grad_step = 000467, loss = 0.001534
<br />grad_step = 000468, loss = 0.001522
<br />grad_step = 000469, loss = 0.001697
<br />grad_step = 000470, loss = 0.001808
<br />grad_step = 000471, loss = 0.001694
<br />grad_step = 000472, loss = 0.001524
<br />grad_step = 000473, loss = 0.001495
<br />grad_step = 000474, loss = 0.001590
<br />grad_step = 000475, loss = 0.001655
<br />grad_step = 000476, loss = 0.001607
<br />grad_step = 000477, loss = 0.001528
<br />grad_step = 000478, loss = 0.001517
<br />grad_step = 000479, loss = 0.001546
<br />grad_step = 000480, loss = 0.001551
<br />grad_step = 000481, loss = 0.001510
<br />grad_step = 000482, loss = 0.001489
<br />grad_step = 000483, loss = 0.001513
<br />grad_step = 000484, loss = 0.001530
<br />grad_step = 000485, loss = 0.001509
<br />grad_step = 000486, loss = 0.001465
<br />grad_step = 000487, loss = 0.001459
<br />grad_step = 000488, loss = 0.001488
<br />grad_step = 000489, loss = 0.001496
<br />grad_step = 000490, loss = 0.001475
<br />grad_step = 000491, loss = 0.001452
<br />grad_step = 000492, loss = 0.001456
<br />grad_step = 000493, loss = 0.001468
<br />grad_step = 000494, loss = 0.001462
<br />grad_step = 000495, loss = 0.001446
<br />grad_step = 000496, loss = 0.001440
<br />grad_step = 000497, loss = 0.001448
<br />grad_step = 000498, loss = 0.001453
<br />grad_step = 000499, loss = 0.001446
<br />grad_step = 000500, loss = 0.001433
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001427
<br />Finished.
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-26 23:57:52.249973
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.217781
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:57:52.257122
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.120736
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:57:52.264934
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.124305
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:57:52.270463
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                 -0.834633
<br />metric_name                                             r2_score
<br />Name: 7, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_fb/fb_prophet/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
<br />Initial log joint probability = -192.039
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />      99       9186.38     0.0272386        1207.2           1           1      123   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     199       10269.2     0.0242289       2566.31        0.89        0.89      233   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     299       10621.2     0.0237499       3262.95           1           1      343   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     399       10886.5     0.0339822       1343.14           1           1      459   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     499       11288.1    0.00255943       1266.79           1           1      580   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     599       11498.7     0.0166167       2146.51           1           1      698   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     699       11555.9     0.0104637       2039.91           1           1      812   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     799       11575.2    0.00955805       570.757           1           1      922   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     899       11630.7     0.0178715       1643.41      0.3435      0.3435     1036   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     999       11700.1      0.034504       2394.16           1           1     1146   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1099       11744.7   0.000237394       144.685           1           1     1258   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1199       11753.1    0.00188838       552.132      0.4814           1     1372   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1299         11758    0.00101299       262.652      0.7415      0.7415     1490   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1399         11761   0.000712302       157.258           1           1     1606   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1499       11781.3     0.0243264       931.457           1           1     1717   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1599       11791.1     0.0025484       550.483      0.7644      0.7644     1834   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1699       11797.7    0.00732868       810.153           1           1     1952   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1799       11802.5   0.000319611       98.1955     0.04871           1     2077   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1818       11803.2   5.97419e-05       246.505   3.588e-07       0.001     2142  LS failed, Hessian reset 
<br />    1855       11803.6   0.000110613       144.447   1.529e-06       0.001     2225  LS failed, Hessian reset 
<br />    1899       11804.3   0.000976631       305.295           1           1     2275   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1999       11805.4   4.67236e-05       72.2243      0.9487      0.9487     2391   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2033       11806.1   1.47341e-05       111.754   8.766e-08       0.001     2480  LS failed, Hessian reset 
<br />    2099       11806.6   9.53816e-05       108.311      0.9684      0.9684     2563   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2151       11806.8   3.32394e-05       152.834   3.931e-07       0.001     2668  LS failed, Hessian reset 
<br />    2199         11807    0.00273479       216.444           1           1     2723   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2299       11810.9    0.00793685       550.165           1           1     2837   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2399       11818.9     0.0134452       377.542           1           1     2952   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2499       11824.9     0.0041384       130.511           1           1     3060   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2525       11826.5   2.36518e-05       102.803   6.403e-08       0.001     3158  LS failed, Hessian reset 
<br />    2599       11827.9   0.000370724       186.394      0.4637      0.4637     3242   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2606         11828   1.70497e-05       123.589     7.9e-08       0.001     3292  LS failed, Hessian reset 
<br />    2699       11829.1    0.00168243       332.201           1           1     3407   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2709       11829.2   1.92694e-05       146.345   1.034e-07       0.001     3461  LS failed, Hessian reset 
<br />    2746       11829.4   1.61976e-05       125.824   9.572e-08       0.001     3551  LS failed, Hessian reset 
<br />    2799       11829.5    0.00491161       122.515           1           1     3615   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2899       11830.6   0.000250007       100.524           1           1     3742   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2999       11830.9    0.00236328       193.309           1           1     3889   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3099       11831.3   0.000309242       194.211      0.7059      0.7059     4015   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3199       11831.4    1.3396e-05       91.8042      0.9217      0.9217     4136   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3299       11831.6   0.000373334       77.3538      0.3184           1     4256   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3399       11831.8   0.000125272       64.7127           1           1     4379   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3499         11832     0.0010491       69.8273           1           1     4503   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3553       11832.1   1.09422e-05       89.3197   8.979e-08       0.001     4612  LS failed, Hessian reset 
<br />    3584       11832.1   8.65844e-07       55.9367      0.4252      0.4252     4658   
<br />Optimization terminated normally: 
<br />  Convergence detected: relative gradient magnitude is below tolerance
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1ea60304e0> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-26 23:58:10.738815
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:58:10.743279
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:58:10.747467
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-26 23:58:10.751305
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -18.2877
<br />metric_name                                             r2_score
<br />Name: 11, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/example/benchmark/ 
<br />
<br />                        date_run  ...            metric_name
<br />0   2020-05-26 23:57:26.454791  ...    mean_absolute_error
<br />1   2020-05-26 23:57:26.459509  ...     mean_squared_error
<br />2   2020-05-26 23:57:26.463913  ...  median_absolute_error
<br />3   2020-05-26 23:57:26.467813  ...               r2_score
<br />4   2020-05-26 23:57:52.249973  ...    mean_absolute_error
<br />5   2020-05-26 23:57:52.257122  ...     mean_squared_error
<br />6   2020-05-26 23:57:52.264934  ...  median_absolute_error
<br />7   2020-05-26 23:57:52.270463  ...               r2_score
<br />8   2020-05-26 23:58:10.738815  ...    mean_absolute_error
<br />9   2020-05-26 23:58:10.743279  ...     mean_squared_error
<br />10  2020-05-26 23:58:10.747467  ...  median_absolute_error
<br />11  2020-05-26 23:58:10.751305  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 10, [Traceback at line 3054](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3054)<br />3054..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'

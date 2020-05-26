## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 665](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L665)<br />665..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 2, [Traceback at line 1845](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1845)<br />1845..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 3, [Traceback at line 1880](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1880)<br />1880..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 4, [Traceback at line 1920](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1920)<br />1920..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 5, [Traceback at line 1955](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1955)<br />1955..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 6, [Traceback at line 2000](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2000)<br />2000..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 7, [Traceback at line 2035](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2035)<br />2035..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 8, [Traceback at line 2092](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2092)<br />2092..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 9, [Traceback at line 2096](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2096)<br />2096..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/ztest/model_keras/armdn/'} 
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
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fae30ffe9e8> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 355447.3125
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 92ms/step - loss: 266818.9375
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 91ms/step - loss: 180637.7969
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 99ms/step - loss: 109401.1094
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 63649.0938
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 94ms/step - loss: 37824.9805
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 90ms/step - loss: 23591.8594
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 92ms/step - loss: 15474.5645
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 100ms/step - loss: 10680.0889
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 7727.3120
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[ 1.2001508   1.9570178   3.1424572   2.5986903   1.6159432   2.68553
<br />   3.3453643   2.2362967   2.4067588   1.7536963   1.8791883   2.4968903
<br />   2.669754    0.6287121   1.811993    2.9511607   1.9249382   2.8003242
<br />   3.8964942   0.66398895  1.4416856   0.41706544  2.6520786   0.87216556
<br />   2.7417605   1.9624082   2.6524339   0.40504515  2.3294857   3.755403
<br />   3.316142    2.6293015   2.5672095   1.6015325   1.475251    2.4262698
<br />   3.5810592   0.5984271   3.7369344   2.6437511   3.874663    3.1105027
<br />   3.937055    2.3189957   2.6964734   1.7878656   3.3599794   1.0563822
<br />   1.8536047   3.5373876   2.6851904   1.7601848   1.9015437   2.2805371
<br />   1.3899517   1.489444    2.993182    2.9796154   2.6310008   2.324724
<br />  -0.26178715  9.353587    6.145688    7.594642    5.477923    8.318471
<br />   7.625364    6.7360973   7.6910152   5.859969    6.9867263   7.6549664
<br />   9.280575    7.1124043   7.4018216   8.286108    7.633419    8.404147
<br />   9.055973    8.486421    7.6629405   6.9065347   7.1163497   5.8349233
<br />   7.4340343   6.9219346   9.182837    7.6494446   7.704532    8.286095
<br />   7.8132873   8.176736    7.263225    7.8223166   8.291272    7.672552
<br />   7.435613    7.57343     6.320163    6.634929    5.898044    7.669543
<br />   7.201299    6.114375    7.0487757   6.786587    5.961644    6.411976
<br />   7.306807    7.8591423   8.75911     7.461471    8.452717    7.351503
<br />   8.637522    7.687986    7.521033    8.305629    6.8767023   6.3902698
<br />  -1.1147547  -2.0296235  -0.10875505 -1.0143806   0.85653424 -0.18652256
<br />  -0.33689934  0.07505056  0.52773905  0.3664744   0.7406107   1.1142719
<br />  -1.4248316  -0.68355745  1.068458   -0.5321584   0.3920786   0.5841378
<br />  -0.7014593  -0.4582693  -1.0956855   1.2560394  -0.97039425 -1.4382169
<br />  -0.02717167  1.0374264   0.5726442   0.46295166 -0.07508849  0.5048327
<br />  -0.47985053 -0.12979925 -1.2729466   0.916075    1.2443259  -1.4542484
<br />   0.44269854 -0.6589302   1.589575    0.12050873 -0.49263903 -0.50002193
<br />  -1.5839291  -1.4052241   0.7701299   0.05962527  0.54165107 -0.48187473
<br />  -0.2887855  -0.28461084  0.51496613  0.44436866  0.3612229  -0.589221
<br />  -0.19846493 -1.2904626   0.20188269  1.2430016  -1.1728914  -0.01709584
<br />   0.10280621  3.6726446   3.0146022   1.9461497   1.8972142   2.8710887
<br />   3.2886038   2.5129356   3.9344769   3.3684344   4.1382723   3.7252498
<br />   3.153006    2.036856    3.3283882   3.4740252   2.0112119   4.0696564
<br />   5.1492624   4.0461864   4.7374487   2.7295551   2.1955466   3.0441647
<br />   4.040676    2.9392242   3.0276523   5.4198174   5.630363    3.100513
<br />   2.9278598   3.6479964   3.7524037   2.1313117   2.6965966   3.8072958
<br />   3.013205    3.2724056   2.5689635   3.4467182   3.3670478   3.215991
<br />   4.0645804   4.5805187   4.853898    3.837882    3.7104173   4.2147865
<br />   2.8156428   4.9721026   2.786896    3.982768    3.1397605   4.551263
<br />   2.981956    5.2645845   4.0205526   3.8505082   3.9646688   1.9407012
<br />   0.1031031   9.6031475   7.000309    7.4965205   8.676715    7.793342
<br />   7.855792    7.0314894   7.931428    6.7771783   7.744169    7.881583
<br />   7.2302456   7.5119967   6.8160133   8.388518    9.909299    7.6721396
<br />   7.855301    6.4545455   8.447994    8.277454    9.676249    8.696967
<br />   8.179631    8.868815    9.410223    8.351894    9.318228    6.9407964
<br />   8.346635    8.113241    7.695943    6.868685    7.7335835   9.744912
<br />   9.060788    7.675575    7.478553    8.874549    6.5423207   7.8982654
<br />   8.475375    7.3878894   6.8388586   8.362974    7.542526    6.922314
<br />   7.597962    7.137532    8.126786    8.309462    8.291766    7.5334806
<br />   8.212679    8.664123    7.1571193   7.7852387   7.516442    8.208283
<br />   0.51322454  0.60076994  0.6932176   2.1560779   1.2490177   1.0135584
<br />   1.1162095   0.6981201   1.6327565   0.22805685  2.1911092   0.709756
<br />   2.3670921   2.3995357   0.27678144  1.1934751   0.92609406  1.0377098
<br />   1.3297029   1.9561298   0.49409074  1.028308    1.2339382   0.4294963
<br />   0.09455681  0.28079128  0.21146822  0.7429472   0.7319597   0.9087072
<br />   0.2267003   0.6228261   0.621002    0.38537467  0.51912796  0.84021926
<br />   1.7754552   0.4462334   2.044631    2.1873782   0.6116891   1.0708696
<br />   1.3888975   0.34969997  1.4737198   0.1200074   0.94617164  0.30286252
<br />   0.70923257  1.4276774   1.5249233   1.2855811   0.5811416   2.4192736
<br />   2.4087963   2.843135    0.51301104  0.20040119  0.59753484  0.47754514
<br />  -8.142208    8.091419   -6.1119413 ]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-25 23:50:00.767821
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   94.6235
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:00.772687
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   8974.13
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:00.776421
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   93.9993
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:00.779872
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -802.696
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
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140385609829288
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328685192
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328685696
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328288952
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328289456
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328289960
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fae45343438> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.587050
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.561916
<br />grad_step = 000002, loss = 0.541793
<br />grad_step = 000003, loss = 0.520126
<br />grad_step = 000004, loss = 0.498513
<br />grad_step = 000005, loss = 0.480177
<br />grad_step = 000006, loss = 0.461525
<br />grad_step = 000007, loss = 0.449246
<br />grad_step = 000008, loss = 0.437747
<br />grad_step = 000009, loss = 0.420483
<br />grad_step = 000010, loss = 0.405302
<br />grad_step = 000011, loss = 0.394317
<br />grad_step = 000012, loss = 0.385426
<br />grad_step = 000013, loss = 0.375614
<br />grad_step = 000014, loss = 0.364080
<br />grad_step = 000015, loss = 0.351789
<br />grad_step = 000016, loss = 0.339817
<br />grad_step = 000017, loss = 0.328966
<br />grad_step = 000018, loss = 0.318169
<br />grad_step = 000019, loss = 0.306596
<br />grad_step = 000020, loss = 0.294172
<br />grad_step = 000021, loss = 0.282086
<br />grad_step = 000022, loss = 0.271267
<br />grad_step = 000023, loss = 0.261464
<br />grad_step = 000024, loss = 0.251714
<br />grad_step = 000025, loss = 0.241488
<br />grad_step = 000026, loss = 0.231311
<br />grad_step = 000027, loss = 0.221924
<br />grad_step = 000028, loss = 0.213180
<br />grad_step = 000029, loss = 0.204352
<br />grad_step = 000030, loss = 0.195210
<br />grad_step = 000031, loss = 0.186251
<br />grad_step = 000032, loss = 0.177978
<br />grad_step = 000033, loss = 0.169987
<br />grad_step = 000034, loss = 0.161910
<br />grad_step = 000035, loss = 0.153957
<br />grad_step = 000036, loss = 0.146228
<br />grad_step = 000037, loss = 0.139027
<br />grad_step = 000038, loss = 0.132073
<br />grad_step = 000039, loss = 0.125203
<br />grad_step = 000040, loss = 0.118656
<br />grad_step = 000041, loss = 0.112483
<br />grad_step = 000042, loss = 0.106463
<br />grad_step = 000043, loss = 0.100513
<br />grad_step = 000044, loss = 0.094787
<br />grad_step = 000045, loss = 0.089434
<br />grad_step = 000046, loss = 0.084300
<br />grad_step = 000047, loss = 0.077838
<br />grad_step = 000048, loss = 0.071586
<br />grad_step = 000049, loss = 0.066198
<br />grad_step = 000050, loss = 0.061874
<br />grad_step = 000051, loss = 0.058092
<br />grad_step = 000052, loss = 0.054448
<br />grad_step = 000053, loss = 0.050751
<br />grad_step = 000054, loss = 0.047056
<br />grad_step = 000055, loss = 0.043377
<br />grad_step = 000056, loss = 0.039942
<br />grad_step = 000057, loss = 0.036720
<br />grad_step = 000058, loss = 0.033768
<br />grad_step = 000059, loss = 0.031137
<br />grad_step = 000060, loss = 0.028758
<br />grad_step = 000061, loss = 0.026568
<br />grad_step = 000062, loss = 0.024454
<br />grad_step = 000063, loss = 0.022416
<br />grad_step = 000064, loss = 0.020453
<br />grad_step = 000065, loss = 0.018620
<br />grad_step = 000066, loss = 0.016574
<br />grad_step = 000067, loss = 0.014980
<br />grad_step = 000068, loss = 0.013861
<br />grad_step = 000069, loss = 0.012813
<br />grad_step = 000070, loss = 0.011740
<br />grad_step = 000071, loss = 0.010700
<br />grad_step = 000072, loss = 0.009776
<br />grad_step = 000073, loss = 0.009012
<br />grad_step = 000074, loss = 0.008275
<br />grad_step = 000075, loss = 0.007559
<br />grad_step = 000076, loss = 0.006949
<br />grad_step = 000077, loss = 0.006420
<br />grad_step = 000078, loss = 0.005987
<br />grad_step = 000079, loss = 0.005591
<br />grad_step = 000080, loss = 0.005203
<br />grad_step = 000081, loss = 0.004854
<br />grad_step = 000082, loss = 0.004519
<br />grad_step = 000083, loss = 0.004210
<br />grad_step = 000084, loss = 0.003958
<br />grad_step = 000085, loss = 0.003743
<br />grad_step = 000086, loss = 0.003555
<br />grad_step = 000087, loss = 0.003365
<br />grad_step = 000088, loss = 0.003190
<br />grad_step = 000089, loss = 0.003056
<br />grad_step = 000090, loss = 0.002941
<br />grad_step = 000091, loss = 0.002832
<br />grad_step = 000092, loss = 0.002727
<br />grad_step = 000093, loss = 0.002632
<br />grad_step = 000094, loss = 0.002565
<br />grad_step = 000095, loss = 0.002507
<br />grad_step = 000096, loss = 0.002447
<br />grad_step = 000097, loss = 0.002388
<br />grad_step = 000098, loss = 0.002335
<br />grad_step = 000099, loss = 0.002298
<br />grad_step = 000100, loss = 0.002268
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002236
<br />grad_step = 000102, loss = 0.002208
<br />grad_step = 000103, loss = 0.002186
<br />grad_step = 000104, loss = 0.002175
<br />grad_step = 000105, loss = 0.002169
<br />grad_step = 000106, loss = 0.002165
<br />grad_step = 000107, loss = 0.002155
<br />grad_step = 000108, loss = 0.002140
<br />grad_step = 000109, loss = 0.002118
<br />grad_step = 000110, loss = 0.002097
<br />grad_step = 000111, loss = 0.002084
<br />grad_step = 000112, loss = 0.002081
<br />grad_step = 000113, loss = 0.002083
<br />grad_step = 000114, loss = 0.002087
<br />grad_step = 000115, loss = 0.002091
<br />grad_step = 000116, loss = 0.002091
<br />grad_step = 000117, loss = 0.002089
<br />grad_step = 000118, loss = 0.002079
<br />grad_step = 000119, loss = 0.002066
<br />grad_step = 000120, loss = 0.002049
<br />grad_step = 000121, loss = 0.002035
<br />grad_step = 000122, loss = 0.002024
<br />grad_step = 000123, loss = 0.002016
<br />grad_step = 000124, loss = 0.002010
<br />grad_step = 000125, loss = 0.002007
<br />grad_step = 000126, loss = 0.002007
<br />grad_step = 000127, loss = 0.002015
<br />grad_step = 000128, loss = 0.002049
<br />grad_step = 000129, loss = 0.002121
<br />grad_step = 000130, loss = 0.002252
<br />grad_step = 000131, loss = 0.002224
<br />grad_step = 000132, loss = 0.002124
<br />grad_step = 000133, loss = 0.002013
<br />grad_step = 000134, loss = 0.002023
<br />grad_step = 000135, loss = 0.002111
<br />grad_step = 000136, loss = 0.002068
<br />grad_step = 000137, loss = 0.001957
<br />grad_step = 000138, loss = 0.002020
<br />grad_step = 000139, loss = 0.002066
<br />grad_step = 000140, loss = 0.001973
<br />grad_step = 000141, loss = 0.001954
<br />grad_step = 000142, loss = 0.002011
<br />grad_step = 000143, loss = 0.001988
<br />grad_step = 000144, loss = 0.001964
<br />grad_step = 000145, loss = 0.001974
<br />grad_step = 000146, loss = 0.001933
<br />grad_step = 000147, loss = 0.001960
<br />grad_step = 000148, loss = 0.001978
<br />grad_step = 000149, loss = 0.001921
<br />grad_step = 000150, loss = 0.001923
<br />grad_step = 000151, loss = 0.001954
<br />grad_step = 000152, loss = 0.001927
<br />grad_step = 000153, loss = 0.001907
<br />grad_step = 000154, loss = 0.001923
<br />grad_step = 000155, loss = 0.001906
<br />grad_step = 000156, loss = 0.001910
<br />grad_step = 000157, loss = 0.001913
<br />grad_step = 000158, loss = 0.001888
<br />grad_step = 000159, loss = 0.001887
<br />grad_step = 000160, loss = 0.001900
<br />grad_step = 000161, loss = 0.001896
<br />grad_step = 000162, loss = 0.001880
<br />grad_step = 000163, loss = 0.001884
<br />grad_step = 000164, loss = 0.001877
<br />grad_step = 000165, loss = 0.001865
<br />grad_step = 000166, loss = 0.001868
<br />grad_step = 000167, loss = 0.001871
<br />grad_step = 000168, loss = 0.001869
<br />grad_step = 000169, loss = 0.001863
<br />grad_step = 000170, loss = 0.001866
<br />grad_step = 000171, loss = 0.001864
<br />grad_step = 000172, loss = 0.001853
<br />grad_step = 000173, loss = 0.001848
<br />grad_step = 000174, loss = 0.001845
<br />grad_step = 000175, loss = 0.001842
<br />grad_step = 000176, loss = 0.001837
<br />grad_step = 000177, loss = 0.001834
<br />grad_step = 000178, loss = 0.001835
<br />grad_step = 000179, loss = 0.001835
<br />grad_step = 000180, loss = 0.001836
<br />grad_step = 000181, loss = 0.001844
<br />grad_step = 000182, loss = 0.001862
<br />grad_step = 000183, loss = 0.001909
<br />grad_step = 000184, loss = 0.001953
<br />grad_step = 000185, loss = 0.002003
<br />grad_step = 000186, loss = 0.001931
<br />grad_step = 000187, loss = 0.001842
<br />grad_step = 000188, loss = 0.001818
<br />grad_step = 000189, loss = 0.001866
<br />grad_step = 000190, loss = 0.001887
<br />grad_step = 000191, loss = 0.001837
<br />grad_step = 000192, loss = 0.001807
<br />grad_step = 000193, loss = 0.001828
<br />grad_step = 000194, loss = 0.001847
<br />grad_step = 000195, loss = 0.001837
<br />grad_step = 000196, loss = 0.001800
<br />grad_step = 000197, loss = 0.001786
<br />grad_step = 000198, loss = 0.001808
<br />grad_step = 000199, loss = 0.001820
<br />grad_step = 000200, loss = 0.001802
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001780
<br />grad_step = 000202, loss = 0.001782
<br />grad_step = 000203, loss = 0.001792
<br />grad_step = 000204, loss = 0.001790
<br />grad_step = 000205, loss = 0.001780
<br />grad_step = 000206, loss = 0.001768
<br />grad_step = 000207, loss = 0.001768
<br />grad_step = 000208, loss = 0.001777
<br />grad_step = 000209, loss = 0.001778
<br />grad_step = 000210, loss = 0.001772
<br />grad_step = 000211, loss = 0.001764
<br />grad_step = 000212, loss = 0.001758
<br />grad_step = 000213, loss = 0.001756
<br />grad_step = 000214, loss = 0.001759
<br />grad_step = 000215, loss = 0.001764
<br />grad_step = 000216, loss = 0.001763
<br />grad_step = 000217, loss = 0.001763
<br />grad_step = 000218, loss = 0.001761
<br />grad_step = 000219, loss = 0.001764
<br />grad_step = 000220, loss = 0.001767
<br />grad_step = 000221, loss = 0.001783
<br />grad_step = 000222, loss = 0.001787
<br />grad_step = 000223, loss = 0.001785
<br />grad_step = 000224, loss = 0.001755
<br />grad_step = 000225, loss = 0.001732
<br />grad_step = 000226, loss = 0.001730
<br />grad_step = 000227, loss = 0.001745
<br />grad_step = 000228, loss = 0.001768
<br />grad_step = 000229, loss = 0.001773
<br />grad_step = 000230, loss = 0.001772
<br />grad_step = 000231, loss = 0.001753
<br />grad_step = 000232, loss = 0.001747
<br />grad_step = 000233, loss = 0.001761
<br />grad_step = 000234, loss = 0.001810
<br />grad_step = 000235, loss = 0.001862
<br />grad_step = 000236, loss = 0.001931
<br />grad_step = 000237, loss = 0.001897
<br />grad_step = 000238, loss = 0.001833
<br />grad_step = 000239, loss = 0.001737
<br />grad_step = 000240, loss = 0.001728
<br />grad_step = 000241, loss = 0.001790
<br />grad_step = 000242, loss = 0.001824
<br />grad_step = 000243, loss = 0.001786
<br />grad_step = 000244, loss = 0.001716
<br />grad_step = 000245, loss = 0.001703
<br />grad_step = 000246, loss = 0.001742
<br />grad_step = 000247, loss = 0.001755
<br />grad_step = 000248, loss = 0.001728
<br />grad_step = 000249, loss = 0.001696
<br />grad_step = 000250, loss = 0.001705
<br />grad_step = 000251, loss = 0.001731
<br />grad_step = 000252, loss = 0.001721
<br />grad_step = 000253, loss = 0.001693
<br />grad_step = 000254, loss = 0.001681
<br />grad_step = 000255, loss = 0.001693
<br />grad_step = 000256, loss = 0.001701
<br />grad_step = 000257, loss = 0.001689
<br />grad_step = 000258, loss = 0.001672
<br />grad_step = 000259, loss = 0.001669
<br />grad_step = 000260, loss = 0.001677
<br />grad_step = 000261, loss = 0.001681
<br />grad_step = 000262, loss = 0.001672
<br />grad_step = 000263, loss = 0.001662
<br />grad_step = 000264, loss = 0.001659
<br />grad_step = 000265, loss = 0.001665
<br />grad_step = 000266, loss = 0.001671
<br />grad_step = 000267, loss = 0.001678
<br />grad_step = 000268, loss = 0.001691
<br />grad_step = 000269, loss = 0.001740
<br />grad_step = 000270, loss = 0.001778
<br />grad_step = 000271, loss = 0.001851
<br />grad_step = 000272, loss = 0.001780
<br />grad_step = 000273, loss = 0.001718
<br />grad_step = 000274, loss = 0.001686
<br />grad_step = 000275, loss = 0.001686
<br />grad_step = 000276, loss = 0.001698
<br />grad_step = 000277, loss = 0.001700
<br />grad_step = 000278, loss = 0.001701
<br />grad_step = 000279, loss = 0.001672
<br />grad_step = 000280, loss = 0.001642
<br />grad_step = 000281, loss = 0.001641
<br />grad_step = 000282, loss = 0.001674
<br />grad_step = 000283, loss = 0.001705
<br />grad_step = 000284, loss = 0.001691
<br />grad_step = 000285, loss = 0.001678
<br />grad_step = 000286, loss = 0.001674
<br />grad_step = 000287, loss = 0.001682
<br />grad_step = 000288, loss = 0.001662
<br />grad_step = 000289, loss = 0.001629
<br />grad_step = 000290, loss = 0.001614
<br />grad_step = 000291, loss = 0.001629
<br />grad_step = 000292, loss = 0.001647
<br />grad_step = 000293, loss = 0.001631
<br />grad_step = 000294, loss = 0.001609
<br />grad_step = 000295, loss = 0.001606
<br />grad_step = 000296, loss = 0.001617
<br />grad_step = 000297, loss = 0.001619
<br />grad_step = 000298, loss = 0.001609
<br />grad_step = 000299, loss = 0.001597
<br />grad_step = 000300, loss = 0.001597
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001604
<br />grad_step = 000302, loss = 0.001603
<br />grad_step = 000303, loss = 0.001595
<br />grad_step = 000304, loss = 0.001587
<br />grad_step = 000305, loss = 0.001585
<br />grad_step = 000306, loss = 0.001588
<br />grad_step = 000307, loss = 0.001589
<br />grad_step = 000308, loss = 0.001588
<br />grad_step = 000309, loss = 0.001583
<br />grad_step = 000310, loss = 0.001579
<br />grad_step = 000311, loss = 0.001581
<br />grad_step = 000312, loss = 0.001590
<br />grad_step = 000313, loss = 0.001605
<br />grad_step = 000314, loss = 0.001640
<br />grad_step = 000315, loss = 0.001674
<br />grad_step = 000316, loss = 0.001758
<br />grad_step = 000317, loss = 0.001768
<br />grad_step = 000318, loss = 0.001809
<br />grad_step = 000319, loss = 0.001703
<br />grad_step = 000320, loss = 0.001615
<br />grad_step = 000321, loss = 0.001583
<br />grad_step = 000322, loss = 0.001622
<br />grad_step = 000323, loss = 0.001680
<br />grad_step = 000324, loss = 0.001654
<br />grad_step = 000325, loss = 0.001603
<br />grad_step = 000326, loss = 0.001558
<br />grad_step = 000327, loss = 0.001576
<br />grad_step = 000328, loss = 0.001622
<br />grad_step = 000329, loss = 0.001614
<br />grad_step = 000330, loss = 0.001582
<br />grad_step = 000331, loss = 0.001553
<br />grad_step = 000332, loss = 0.001558
<br />grad_step = 000333, loss = 0.001572
<br />grad_step = 000334, loss = 0.001566
<br />grad_step = 000335, loss = 0.001560
<br />grad_step = 000336, loss = 0.001564
<br />grad_step = 000337, loss = 0.001575
<br />grad_step = 000338, loss = 0.001566
<br />grad_step = 000339, loss = 0.001549
<br />grad_step = 000340, loss = 0.001535
<br />grad_step = 000341, loss = 0.001534
<br />grad_step = 000342, loss = 0.001538
<br />grad_step = 000343, loss = 0.001538
<br />grad_step = 000344, loss = 0.001533
<br />grad_step = 000345, loss = 0.001531
<br />grad_step = 000346, loss = 0.001535
<br />grad_step = 000347, loss = 0.001539
<br />grad_step = 000348, loss = 0.001540
<br />grad_step = 000349, loss = 0.001536
<br />grad_step = 000350, loss = 0.001533
<br />grad_step = 000351, loss = 0.001532
<br />grad_step = 000352, loss = 0.001534
<br />grad_step = 000353, loss = 0.001530
<br />grad_step = 000354, loss = 0.001527
<br />grad_step = 000355, loss = 0.001521
<br />grad_step = 000356, loss = 0.001520
<br />grad_step = 000357, loss = 0.001519
<br />grad_step = 000358, loss = 0.001520
<br />grad_step = 000359, loss = 0.001517
<br />grad_step = 000360, loss = 0.001515
<br />grad_step = 000361, loss = 0.001514
<br />grad_step = 000362, loss = 0.001516
<br />grad_step = 000363, loss = 0.001518
<br />grad_step = 000364, loss = 0.001525
<br />grad_step = 000365, loss = 0.001530
<br />grad_step = 000366, loss = 0.001544
<br />grad_step = 000367, loss = 0.001554
<br />grad_step = 000368, loss = 0.001579
<br />grad_step = 000369, loss = 0.001584
<br />grad_step = 000370, loss = 0.001596
<br />grad_step = 000371, loss = 0.001568
<br />grad_step = 000372, loss = 0.001539
<br />grad_step = 000373, loss = 0.001502
<br />grad_step = 000374, loss = 0.001485
<br />grad_step = 000375, loss = 0.001488
<br />grad_step = 000376, loss = 0.001505
<br />grad_step = 000377, loss = 0.001527
<br />grad_step = 000378, loss = 0.001540
<br />grad_step = 000379, loss = 0.001555
<br />grad_step = 000380, loss = 0.001542
<br />grad_step = 000381, loss = 0.001527
<br />grad_step = 000382, loss = 0.001497
<br />grad_step = 000383, loss = 0.001476
<br />grad_step = 000384, loss = 0.001469
<br />grad_step = 000385, loss = 0.001474
<br />grad_step = 000386, loss = 0.001487
<br />grad_step = 000387, loss = 0.001498
<br />grad_step = 000388, loss = 0.001514
<br />grad_step = 000389, loss = 0.001521
<br />grad_step = 000390, loss = 0.001536
<br />grad_step = 000391, loss = 0.001531
<br />grad_step = 000392, loss = 0.001520
<br />grad_step = 000393, loss = 0.001490
<br />grad_step = 000394, loss = 0.001465
<br />grad_step = 000395, loss = 0.001453
<br />grad_step = 000396, loss = 0.001458
<br />grad_step = 000397, loss = 0.001471
<br />grad_step = 000398, loss = 0.001483
<br />grad_step = 000399, loss = 0.001496
<br />grad_step = 000400, loss = 0.001497
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001500
<br />grad_step = 000402, loss = 0.001489
<br />grad_step = 000403, loss = 0.001477
<br />grad_step = 000404, loss = 0.001460
<br />grad_step = 000405, loss = 0.001448
<br />grad_step = 000406, loss = 0.001441
<br />grad_step = 000407, loss = 0.001441
<br />grad_step = 000408, loss = 0.001448
<br />grad_step = 000409, loss = 0.001457
<br />grad_step = 000410, loss = 0.001470
<br />grad_step = 000411, loss = 0.001479
<br />grad_step = 000412, loss = 0.001497
<br />grad_step = 000413, loss = 0.001502
<br />grad_step = 000414, loss = 0.001511
<br />grad_step = 000415, loss = 0.001498
<br />grad_step = 000416, loss = 0.001485
<br />grad_step = 000417, loss = 0.001457
<br />grad_step = 000418, loss = 0.001435
<br />grad_step = 000419, loss = 0.001421
<br />grad_step = 000420, loss = 0.001421
<br />grad_step = 000421, loss = 0.001431
<br />grad_step = 000422, loss = 0.001444
<br />grad_step = 000423, loss = 0.001463
<br />grad_step = 000424, loss = 0.001469
<br />grad_step = 000425, loss = 0.001476
<br />grad_step = 000426, loss = 0.001462
<br />grad_step = 000427, loss = 0.001448
<br />grad_step = 000428, loss = 0.001429
<br />grad_step = 000429, loss = 0.001420
<br />grad_step = 000430, loss = 0.001416
<br />grad_step = 000431, loss = 0.001414
<br />grad_step = 000432, loss = 0.001413
<br />grad_step = 000433, loss = 0.001414
<br />grad_step = 000434, loss = 0.001423
<br />grad_step = 000435, loss = 0.001433
<br />grad_step = 000436, loss = 0.001455
<br />grad_step = 000437, loss = 0.001469
<br />grad_step = 000438, loss = 0.001486
<br />grad_step = 000439, loss = 0.001471
<br />grad_step = 000440, loss = 0.001452
<br />grad_step = 000441, loss = 0.001420
<br />grad_step = 000442, loss = 0.001400
<br />grad_step = 000443, loss = 0.001393
<br />grad_step = 000444, loss = 0.001397
<br />grad_step = 000445, loss = 0.001404
<br />grad_step = 000446, loss = 0.001408
<br />grad_step = 000447, loss = 0.001412
<br />grad_step = 000448, loss = 0.001413
<br />grad_step = 000449, loss = 0.001420
<br />grad_step = 000450, loss = 0.001421
<br />grad_step = 000451, loss = 0.001421
<br />grad_step = 000452, loss = 0.001405
<br />grad_step = 000453, loss = 0.001389
<br />grad_step = 000454, loss = 0.001373
<br />grad_step = 000455, loss = 0.001367
<br />grad_step = 000456, loss = 0.001368
<br />grad_step = 000457, loss = 0.001374
<br />grad_step = 000458, loss = 0.001386
<br />grad_step = 000459, loss = 0.001397
<br />grad_step = 000460, loss = 0.001417
<br />grad_step = 000461, loss = 0.001428
<br />grad_step = 000462, loss = 0.001449
<br />grad_step = 000463, loss = 0.001452
<br />grad_step = 000464, loss = 0.001470
<br />grad_step = 000465, loss = 0.001461
<br />grad_step = 000466, loss = 0.001456
<br />grad_step = 000467, loss = 0.001415
<br />grad_step = 000468, loss = 0.001379
<br />grad_step = 000469, loss = 0.001361
<br />grad_step = 000470, loss = 0.001364
<br />grad_step = 000471, loss = 0.001381
<br />grad_step = 000472, loss = 0.001393
<br />grad_step = 000473, loss = 0.001403
<br />grad_step = 000474, loss = 0.001383
<br />grad_step = 000475, loss = 0.001357
<br />grad_step = 000476, loss = 0.001337
<br />grad_step = 000477, loss = 0.001341
<br />grad_step = 000478, loss = 0.001355
<br />grad_step = 000479, loss = 0.001356
<br />grad_step = 000480, loss = 0.001347
<br />grad_step = 000481, loss = 0.001340
<br />grad_step = 000482, loss = 0.001343
<br />grad_step = 000483, loss = 0.001344
<br />grad_step = 000484, loss = 0.001345
<br />grad_step = 000485, loss = 0.001349
<br />grad_step = 000486, loss = 0.001365
<br />grad_step = 000487, loss = 0.001377
<br />grad_step = 000488, loss = 0.001391
<br />grad_step = 000489, loss = 0.001390
<br />grad_step = 000490, loss = 0.001398
<br />grad_step = 000491, loss = 0.001388
<br />grad_step = 000492, loss = 0.001387
<br />grad_step = 000493, loss = 0.001367
<br />grad_step = 000494, loss = 0.001358
<br />grad_step = 000495, loss = 0.001346
<br />grad_step = 000496, loss = 0.001343
<br />grad_step = 000497, loss = 0.001335
<br />grad_step = 000498, loss = 0.001325
<br />grad_step = 000499, loss = 0.001316
<br />grad_step = 000500, loss = 0.001310
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001310
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
<br />  date_run                              2020-05-25 23:50:21.820379
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.215634
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:21.826121
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.106755
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:21.832685
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.137337
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:21.838868
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                 -0.622173
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
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/ztest/model_fb/fb_prophet/'} 
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
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fae1267c860> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-25 23:50:38.315994
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:38.319588
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:38.323475
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-25 23:50:38.327122
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
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/example/benchmark/ 
<br />
<br />                        date_run  ...            metric_name
<br />0   2020-05-25 23:50:00.767821  ...    mean_absolute_error
<br />1   2020-05-25 23:50:00.772687  ...     mean_squared_error
<br />2   2020-05-25 23:50:00.776421  ...  median_absolute_error
<br />3   2020-05-25 23:50:00.779872  ...               r2_score
<br />4   2020-05-25 23:50:21.820379  ...    mean_absolute_error
<br />5   2020-05-25 23:50:21.826121  ...     mean_squared_error
<br />6   2020-05-25 23:50:21.832685  ...  median_absolute_error
<br />7   2020-05-25 23:50:21.838868  ...               r2_score
<br />8   2020-05-25 23:50:38.315994  ...    mean_absolute_error
<br />9   2020-05-25 23:50:38.319588  ...     mean_squared_error
<br />10  2020-05-25 23:50:38.323475  ...  median_absolute_error
<br />11  2020-05-25 23:50:38.327122  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 10, [Traceback at line 3054](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3054)<br />3054..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'

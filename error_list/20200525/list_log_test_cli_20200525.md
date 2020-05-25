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
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fadf9fe2ef0> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 355159.3438
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 105ms/step - loss: 254631.9219
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 149965.3125
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 123ms/step - loss: 69603.4141
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 98ms/step - loss: 33129.3984
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 96ms/step - loss: 18093.0547
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 97ms/step - loss: 11140.3555
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 90ms/step - loss: 7520.6914
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 96ms/step - loss: 5476.1899
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 4244.0474
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[ 7.58241236e-01  1.00474930e+01  8.58631516e+00  8.28756237e+00
<br />   1.07363319e+01  1.10237722e+01  1.01954842e+01  1.19974575e+01
<br />   1.02631092e+01  1.08429861e+01  9.71825123e+00  1.23190956e+01
<br />   1.03930454e+01  1.03556147e+01  9.58613873e+00  9.75504303e+00
<br />   9.51171398e+00  1.06540012e+01  1.16401415e+01  8.92627048e+00
<br />   1.03619499e+01  1.00662251e+01  1.04804764e+01  1.16749897e+01
<br />   1.14299002e+01  1.08704624e+01  9.63304615e+00  1.12094498e+01
<br />   1.12739305e+01  1.06485205e+01  1.16434851e+01  1.15409822e+01
<br />   1.05735712e+01  1.03733549e+01  1.24620571e+01  1.08848181e+01
<br />   1.05585346e+01  1.03585634e+01  8.18815041e+00  9.75719738e+00
<br />   1.10162687e+01  9.81094933e+00  8.52575207e+00  1.12402344e+01
<br />   1.12881069e+01  9.41869450e+00  1.02830753e+01  1.10104246e+01
<br />   1.19105206e+01  1.03413849e+01  1.16141424e+01  1.11168051e+01
<br />   1.19663343e+01  1.06491795e+01  1.23195705e+01  9.21172810e+00
<br />   1.15085735e+01  1.22995043e+01  9.15082741e+00  8.33281612e+00
<br />   1.05767429e+00  5.35720825e-01 -1.23336005e+00  6.71170533e-01
<br />   5.35058081e-01 -1.44796252e+00  1.05740666e-01 -1.53557062e+00
<br />  -8.17293525e-01 -7.97613561e-01  1.45518100e+00 -2.00497293e+00
<br />   7.60752797e-01 -4.44297463e-01  3.96611914e-02 -2.59984016e-01
<br />  -2.87828875e+00  1.19433320e+00  6.52335763e-01 -2.12763619e+00
<br />   4.98559117e-01  2.11275339e+00 -1.15015638e+00 -7.58543015e-01
<br />  -4.15328383e-01  1.95329106e+00  7.59741664e-02  1.21132195e-01
<br />   3.27013023e-02 -6.58090472e-01 -3.79569113e-01  3.10168535e-01
<br />   1.12337852e+00  4.51684892e-01 -2.07399583e+00 -2.59040689e+00
<br />  -1.08172178e-01 -1.17318964e+00 -1.31538510e+00  6.73461795e-01
<br />   3.62629116e-01  1.18733668e+00 -2.03127551e+00  4.42589104e-01
<br />  -2.74532294e+00  2.99029976e-01 -4.21253622e-01  4.08789039e-01
<br />   2.83518970e-01  1.58620048e+00  1.27533817e+00 -2.86138368e+00
<br />  -4.36971188e-01 -3.25132251e-01  1.26318741e+00  1.99838495e+00
<br />   1.35874331e+00  1.31342649e+00 -1.80389225e+00 -1.41349643e-01
<br />   1.62292290e+00  4.14418578e-02 -2.20659447e+00 -4.96412337e-01
<br />   1.28560710e+00  1.55887258e+00  6.95712030e-01  1.22685122e+00
<br />  -6.55932009e-01  2.91936517e-01  1.10033178e+00  5.34958303e-01
<br />   1.65645629e-01  3.73533517e-02 -1.15089452e+00  5.53817868e-01
<br />   3.26676846e-01  6.56691492e-01  2.67111611e+00  7.65699387e-01
<br />   1.05245507e+00 -1.24764200e-02  4.25621778e-01 -7.11140394e-01
<br />   3.06997120e-01  1.07459259e+00  1.15023673e-01  4.20615017e-01
<br />   1.53063321e+00  5.04981220e-01 -3.72798800e-01  1.60106826e+00
<br />  -1.34615934e+00 -1.04530156e+00 -1.50864029e+00 -5.80288529e-01
<br />   1.15972471e+00  9.45361614e-01  1.49709320e+00  1.20632482e+00
<br />   2.39588737e-01  9.79952991e-01  8.57984304e-01  1.65162563e+00
<br />  -2.43692130e-01 -5.44475853e-01  1.04109907e+00 -1.06775105e-01
<br />   1.40435410e+00  8.10143650e-01  1.07438874e+00  9.05665815e-01
<br />  -8.11538100e-03  7.89394379e-02 -3.51867080e-02  1.87916231e+00
<br />   7.34012485e-01  6.90618932e-01 -2.14371872e+00 -4.43495095e-01
<br />   5.59347808e-01  1.02639685e+01  8.47668934e+00  1.28643312e+01
<br />   9.11203480e+00  1.00661182e+01  1.09180727e+01  1.12446222e+01
<br />   1.16087208e+01  9.58287048e+00  1.09429092e+01  1.03207226e+01
<br />   1.07105751e+01  1.23705750e+01  1.00280838e+01  1.12182093e+01
<br />   8.23085499e+00  1.19493446e+01  9.54344463e+00  1.06621246e+01
<br />   1.07557125e+01  1.01473942e+01  8.99940491e+00  1.06641283e+01
<br />   1.00920010e+01  1.16750059e+01  1.07213955e+01  1.16015892e+01
<br />   9.52057362e+00  1.19284277e+01  9.48011875e+00  1.16728449e+01
<br />   1.24892349e+01  9.57375717e+00  1.10168743e+01  1.25381212e+01
<br />   1.25771179e+01  8.57304764e+00  1.11344728e+01  1.09794016e+01
<br />   7.22049618e+00  9.38496780e+00  1.12940397e+01  1.13727760e+01
<br />   1.00391150e+01  1.06867962e+01  1.08547974e+01  1.15774374e+01
<br />   1.08296547e+01  9.97325897e+00  8.34868240e+00  1.03506031e+01
<br />   1.14250660e+01  1.04032898e+01  1.03991299e+01  1.14823914e+01
<br />   1.10827360e+01  1.06450348e+01  1.23603506e+01  1.21311255e+01
<br />   2.51233876e-01  1.10382617e-01  1.05965638e+00  3.39905977e+00
<br />   2.00107098e-01  2.69608450e+00  1.04518545e+00  1.02941990e+00
<br />   3.81949663e-01  3.23044348e+00  5.34855127e-02  5.37047327e-01
<br />   2.07081556e-01  3.47299039e-01  2.25259662e-01  4.04028225e+00
<br />   1.63387132e+00  3.27896070e+00  7.52836108e-01  1.73452151e+00
<br />   3.03526521e-01  1.96390033e-01  3.38096023e-02  2.29405403e+00
<br />   5.51586449e-01  3.28032076e-01  5.02126396e-01  1.81043100e+00
<br />   4.73028064e-01  2.41983867e+00  2.21140087e-01  1.33261418e+00
<br />   3.50891531e-01  2.18965530e+00  1.57464600e+00  1.50206256e+00
<br />   2.26130152e+00  2.11918414e-01  4.36107159e-01  5.52569866e-01
<br />   1.89825809e+00  2.20164418e-01  3.25451279e+00  1.75449085e+00
<br />   4.77095008e-01  2.42477596e-01  8.96132767e-01  2.61025047e+00
<br />   1.06030536e+00  1.91480350e+00  1.84638262e-01  4.78353918e-01
<br />   6.18922770e-01  9.77392197e-02  8.28112841e-01  7.47855365e-01
<br />   1.27590942e+00  4.80714858e-01  2.37146854e-01  3.91427159e-01
<br />   3.05290556e+00  1.53665185e+00  2.94672072e-01  3.34888935e-01
<br />   8.53613853e-01  3.73989105e-01  7.69696474e-01  5.66570401e-01
<br />   2.07964897e+00  4.44297612e-01  1.11395597e-01  1.10858846e+00
<br />   7.40005374e-02  5.01876593e-01  5.57609141e-01  1.85073781e+00
<br />   2.67482424e+00  2.32530260e+00  1.71122873e+00  2.79887390e+00
<br />   2.55586505e-01  3.89538527e-01  5.22809684e-01  2.57085013e+00
<br />   1.78458452e-01  3.70021999e-01  5.09395659e-01  7.36777961e-01
<br />   8.67535114e-01  3.44021857e-01  2.21316040e-01  2.17744970e+00
<br />   3.33835006e-01  1.37909615e+00  2.43881464e+00  1.10485506e+00
<br />   1.24785411e+00  2.92233038e+00  2.61524868e+00  2.67013311e+00
<br />   3.61222029e-01  2.88335657e+00  1.61220765e+00  1.03120399e+00
<br />   4.63697433e-01  2.78601170e-01  3.95328760e-01  5.68162858e-01
<br />   1.98485184e+00  1.39184666e+00  1.48609030e+00  1.92544127e+00
<br />   2.02802610e+00  1.45900893e+00  1.35532808e+00  1.39993834e+00
<br />   1.11272216e+00  8.03375006e-01  3.41547668e-01  1.21652758e+00
<br />   8.60133266e+00 -6.90220356e+00 -5.67612171e+00]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-24 23:56:53.492233
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                    91.114
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:56:53.499298
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   8334.62
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:56:53.504516
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   91.2444
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:56:53.509338
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -745.424
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
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140385307162328
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405934272
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405934776
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405529840
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405530344
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405530848
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fadf9fd9080> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.543901
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.509596
<br />grad_step = 000002, loss = 0.486128
<br />grad_step = 000003, loss = 0.461945
<br />grad_step = 000004, loss = 0.434883
<br />grad_step = 000005, loss = 0.405255
<br />grad_step = 000006, loss = 0.375870
<br />grad_step = 000007, loss = 0.351548
<br />grad_step = 000008, loss = 0.335632
<br />grad_step = 000009, loss = 0.317571
<br />grad_step = 000010, loss = 0.297870
<br />grad_step = 000011, loss = 0.280454
<br />grad_step = 000012, loss = 0.267626
<br />grad_step = 000013, loss = 0.257748
<br />grad_step = 000014, loss = 0.248483
<br />grad_step = 000015, loss = 0.238800
<br />grad_step = 000016, loss = 0.228886
<br />grad_step = 000017, loss = 0.218523
<br />grad_step = 000018, loss = 0.207786
<br />grad_step = 000019, loss = 0.197174
<br />grad_step = 000020, loss = 0.187318
<br />grad_step = 000021, loss = 0.178344
<br />grad_step = 000022, loss = 0.169370
<br />grad_step = 000023, loss = 0.160377
<br />grad_step = 000024, loss = 0.151878
<br />grad_step = 000025, loss = 0.143956
<br />grad_step = 000026, loss = 0.136446
<br />grad_step = 000027, loss = 0.129055
<br />grad_step = 000028, loss = 0.121939
<br />grad_step = 000029, loss = 0.115279
<br />grad_step = 000030, loss = 0.108860
<br />grad_step = 000031, loss = 0.102476
<br />grad_step = 000032, loss = 0.096147
<br />grad_step = 000033, loss = 0.090252
<br />grad_step = 000034, loss = 0.084812
<br />grad_step = 000035, loss = 0.079522
<br />grad_step = 000036, loss = 0.074395
<br />grad_step = 000037, loss = 0.069611
<br />grad_step = 000038, loss = 0.065209
<br />grad_step = 000039, loss = 0.060989
<br />grad_step = 000040, loss = 0.056886
<br />grad_step = 000041, loss = 0.053009
<br />grad_step = 000042, loss = 0.048856
<br />grad_step = 000043, loss = 0.044285
<br />grad_step = 000044, loss = 0.040204
<br />grad_step = 000045, loss = 0.037310
<br />grad_step = 000046, loss = 0.034842
<br />grad_step = 000047, loss = 0.032250
<br />grad_step = 000048, loss = 0.029536
<br />grad_step = 000049, loss = 0.026832
<br />grad_step = 000050, loss = 0.024463
<br />grad_step = 000051, loss = 0.022438
<br />grad_step = 000052, loss = 0.020457
<br />grad_step = 000053, loss = 0.018768
<br />grad_step = 000054, loss = 0.017197
<br />grad_step = 000055, loss = 0.015583
<br />grad_step = 000056, loss = 0.014118
<br />grad_step = 000057, loss = 0.012792
<br />grad_step = 000058, loss = 0.011639
<br />grad_step = 000059, loss = 0.010649
<br />grad_step = 000060, loss = 0.009671
<br />grad_step = 000061, loss = 0.008809
<br />grad_step = 000062, loss = 0.008020
<br />grad_step = 000063, loss = 0.007280
<br />grad_step = 000064, loss = 0.006658
<br />grad_step = 000065, loss = 0.006059
<br />grad_step = 000066, loss = 0.005526
<br />grad_step = 000067, loss = 0.005086
<br />grad_step = 000068, loss = 0.004690
<br />grad_step = 000069, loss = 0.004330
<br />grad_step = 000070, loss = 0.003987
<br />grad_step = 000071, loss = 0.003689
<br />grad_step = 000072, loss = 0.003448
<br />grad_step = 000073, loss = 0.003218
<br />grad_step = 000074, loss = 0.003034
<br />grad_step = 000075, loss = 0.002869
<br />grad_step = 000076, loss = 0.002728
<br />grad_step = 000077, loss = 0.002614
<br />grad_step = 000078, loss = 0.002500
<br />grad_step = 000079, loss = 0.002404
<br />grad_step = 000080, loss = 0.002327
<br />grad_step = 000081, loss = 0.002278
<br />grad_step = 000082, loss = 0.002235
<br />grad_step = 000083, loss = 0.002189
<br />grad_step = 000084, loss = 0.002153
<br />grad_step = 000085, loss = 0.002122
<br />grad_step = 000086, loss = 0.002104
<br />grad_step = 000087, loss = 0.002090
<br />grad_step = 000088, loss = 0.002078
<br />grad_step = 000089, loss = 0.002068
<br />grad_step = 000090, loss = 0.002059
<br />grad_step = 000091, loss = 0.002050
<br />grad_step = 000092, loss = 0.002040
<br />grad_step = 000093, loss = 0.002035
<br />grad_step = 000094, loss = 0.002030
<br />grad_step = 000095, loss = 0.002024
<br />grad_step = 000096, loss = 0.002017
<br />grad_step = 000097, loss = 0.002009
<br />grad_step = 000098, loss = 0.002006
<br />grad_step = 000099, loss = 0.002009
<br />grad_step = 000100, loss = 0.002029
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002056
<br />grad_step = 000102, loss = 0.002063
<br />grad_step = 000103, loss = 0.001998
<br />grad_step = 000104, loss = 0.001955
<br />grad_step = 000105, loss = 0.001977
<br />grad_step = 000106, loss = 0.001994
<br />grad_step = 000107, loss = 0.001959
<br />grad_step = 000108, loss = 0.001923
<br />grad_step = 000109, loss = 0.001934
<br />grad_step = 000110, loss = 0.001949
<br />grad_step = 000111, loss = 0.001923
<br />grad_step = 000112, loss = 0.001892
<br />grad_step = 000113, loss = 0.001892
<br />grad_step = 000114, loss = 0.001905
<br />grad_step = 000115, loss = 0.001900
<br />grad_step = 000116, loss = 0.001874
<br />grad_step = 000117, loss = 0.001856
<br />grad_step = 000118, loss = 0.001859
<br />grad_step = 000119, loss = 0.001865
<br />grad_step = 000120, loss = 0.001858
<br />grad_step = 000121, loss = 0.001840
<br />grad_step = 000122, loss = 0.001830
<br />grad_step = 000123, loss = 0.001828
<br />grad_step = 000124, loss = 0.001829
<br />grad_step = 000125, loss = 0.001829
<br />grad_step = 000126, loss = 0.001827
<br />grad_step = 000127, loss = 0.001818
<br />grad_step = 000128, loss = 0.001807
<br />grad_step = 000129, loss = 0.001798
<br />grad_step = 000130, loss = 0.001794
<br />grad_step = 000131, loss = 0.001793
<br />grad_step = 000132, loss = 0.001792
<br />grad_step = 000133, loss = 0.001793
<br />grad_step = 000134, loss = 0.001798
<br />grad_step = 000135, loss = 0.001808
<br />grad_step = 000136, loss = 0.001820
<br />grad_step = 000137, loss = 0.001832
<br />grad_step = 000138, loss = 0.001828
<br />grad_step = 000139, loss = 0.001809
<br />grad_step = 000140, loss = 0.001774
<br />grad_step = 000141, loss = 0.001756
<br />grad_step = 000142, loss = 0.001761
<br />grad_step = 000143, loss = 0.001776
<br />grad_step = 000144, loss = 0.001789
<br />grad_step = 000145, loss = 0.001787
<br />grad_step = 000146, loss = 0.001769
<br />grad_step = 000147, loss = 0.001746
<br />grad_step = 000148, loss = 0.001731
<br />grad_step = 000149, loss = 0.001728
<br />grad_step = 000150, loss = 0.001734
<br />grad_step = 000151, loss = 0.001742
<br />grad_step = 000152, loss = 0.001749
<br />grad_step = 000153, loss = 0.001755
<br />grad_step = 000154, loss = 0.001745
<br />grad_step = 000155, loss = 0.001732
<br />grad_step = 000156, loss = 0.001710
<br />grad_step = 000157, loss = 0.001699
<br />grad_step = 000158, loss = 0.001700
<br />grad_step = 000159, loss = 0.001701
<br />grad_step = 000160, loss = 0.001703
<br />grad_step = 000161, loss = 0.001704
<br />grad_step = 000162, loss = 0.001712
<br />grad_step = 000163, loss = 0.001726
<br />grad_step = 000164, loss = 0.001720
<br />grad_step = 000165, loss = 0.001706
<br />grad_step = 000166, loss = 0.001683
<br />grad_step = 000167, loss = 0.001675
<br />grad_step = 000168, loss = 0.001683
<br />grad_step = 000169, loss = 0.001694
<br />grad_step = 000170, loss = 0.001712
<br />grad_step = 000171, loss = 0.001692
<br />grad_step = 000172, loss = 0.001668
<br />grad_step = 000173, loss = 0.001648
<br />grad_step = 000174, loss = 0.001649
<br />grad_step = 000175, loss = 0.001663
<br />grad_step = 000176, loss = 0.001663
<br />grad_step = 000177, loss = 0.001656
<br />grad_step = 000178, loss = 0.001648
<br />grad_step = 000179, loss = 0.001651
<br />grad_step = 000180, loss = 0.001662
<br />grad_step = 000181, loss = 0.001683
<br />grad_step = 000182, loss = 0.001704
<br />grad_step = 000183, loss = 0.001699
<br />grad_step = 000184, loss = 0.001677
<br />grad_step = 000185, loss = 0.001640
<br />grad_step = 000186, loss = 0.001622
<br />grad_step = 000187, loss = 0.001631
<br />grad_step = 000188, loss = 0.001641
<br />grad_step = 000189, loss = 0.001645
<br />grad_step = 000190, loss = 0.001638
<br />grad_step = 000191, loss = 0.001630
<br />grad_step = 000192, loss = 0.001628
<br />grad_step = 000193, loss = 0.001622
<br />grad_step = 000194, loss = 0.001620
<br />grad_step = 000195, loss = 0.001616
<br />grad_step = 000196, loss = 0.001614
<br />grad_step = 000197, loss = 0.001613
<br />grad_step = 000198, loss = 0.001614
<br />grad_step = 000199, loss = 0.001616
<br />grad_step = 000200, loss = 0.001614
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001610
<br />grad_step = 000202, loss = 0.001602
<br />grad_step = 000203, loss = 0.001595
<br />grad_step = 000204, loss = 0.001589
<br />grad_step = 000205, loss = 0.001587
<br />grad_step = 000206, loss = 0.001586
<br />grad_step = 000207, loss = 0.001587
<br />grad_step = 000208, loss = 0.001589
<br />grad_step = 000209, loss = 0.001592
<br />grad_step = 000210, loss = 0.001596
<br />grad_step = 000211, loss = 0.001600
<br />grad_step = 000212, loss = 0.001607
<br />grad_step = 000213, loss = 0.001613
<br />grad_step = 000214, loss = 0.001625
<br />grad_step = 000215, loss = 0.001632
<br />grad_step = 000216, loss = 0.001658
<br />grad_step = 000217, loss = 0.001646
<br />grad_step = 000218, loss = 0.001630
<br />grad_step = 000219, loss = 0.001580
<br />grad_step = 000220, loss = 0.001584
<br />grad_step = 000221, loss = 0.001623
<br />grad_step = 000222, loss = 0.001614
<br />grad_step = 000223, loss = 0.001586
<br />grad_step = 000224, loss = 0.001577
<br />grad_step = 000225, loss = 0.001594
<br />grad_step = 000226, loss = 0.001615
<br />grad_step = 000227, loss = 0.001598
<br />grad_step = 000228, loss = 0.001576
<br />grad_step = 000229, loss = 0.001554
<br />grad_step = 000230, loss = 0.001555
<br />grad_step = 000231, loss = 0.001570
<br />grad_step = 000232, loss = 0.001571
<br />grad_step = 000233, loss = 0.001560
<br />grad_step = 000234, loss = 0.001548
<br />grad_step = 000235, loss = 0.001551
<br />grad_step = 000236, loss = 0.001564
<br />grad_step = 000237, loss = 0.001569
<br />grad_step = 000238, loss = 0.001571
<br />grad_step = 000239, loss = 0.001575
<br />grad_step = 000240, loss = 0.001603
<br />grad_step = 000241, loss = 0.001636
<br />grad_step = 000242, loss = 0.001671
<br />grad_step = 000243, loss = 0.001645
<br />grad_step = 000244, loss = 0.001589
<br />grad_step = 000245, loss = 0.001537
<br />grad_step = 000246, loss = 0.001547
<br />grad_step = 000247, loss = 0.001589
<br />grad_step = 000248, loss = 0.001590
<br />grad_step = 000249, loss = 0.001558
<br />grad_step = 000250, loss = 0.001529
<br />grad_step = 000251, loss = 0.001538
<br />grad_step = 000252, loss = 0.001563
<br />grad_step = 000253, loss = 0.001561
<br />grad_step = 000254, loss = 0.001539
<br />grad_step = 000255, loss = 0.001519
<br />grad_step = 000256, loss = 0.001522
<br />grad_step = 000257, loss = 0.001535
<br />grad_step = 000258, loss = 0.001537
<br />grad_step = 000259, loss = 0.001526
<br />grad_step = 000260, loss = 0.001513
<br />grad_step = 000261, loss = 0.001509
<br />grad_step = 000262, loss = 0.001515
<br />grad_step = 000263, loss = 0.001522
<br />grad_step = 000264, loss = 0.001523
<br />grad_step = 000265, loss = 0.001518
<br />grad_step = 000266, loss = 0.001514
<br />grad_step = 000267, loss = 0.001520
<br />grad_step = 000268, loss = 0.001541
<br />grad_step = 000269, loss = 0.001599
<br />grad_step = 000270, loss = 0.001640
<br />grad_step = 000271, loss = 0.001716
<br />grad_step = 000272, loss = 0.001619
<br />grad_step = 000273, loss = 0.001565
<br />grad_step = 000274, loss = 0.001568
<br />grad_step = 000275, loss = 0.001537
<br />grad_step = 000276, loss = 0.001522
<br />grad_step = 000277, loss = 0.001554
<br />grad_step = 000278, loss = 0.001569
<br />grad_step = 000279, loss = 0.001542
<br />grad_step = 000280, loss = 0.001508
<br />grad_step = 000281, loss = 0.001520
<br />grad_step = 000282, loss = 0.001539
<br />grad_step = 000283, loss = 0.001531
<br />grad_step = 000284, loss = 0.001493
<br />grad_step = 000285, loss = 0.001480
<br />grad_step = 000286, loss = 0.001503
<br />grad_step = 000287, loss = 0.001515
<br />grad_step = 000288, loss = 0.001503
<br />grad_step = 000289, loss = 0.001483
<br />grad_step = 000290, loss = 0.001486
<br />grad_step = 000291, loss = 0.001497
<br />grad_step = 000292, loss = 0.001491
<br />grad_step = 000293, loss = 0.001474
<br />grad_step = 000294, loss = 0.001467
<br />grad_step = 000295, loss = 0.001473
<br />grad_step = 000296, loss = 0.001480
<br />grad_step = 000297, loss = 0.001474
<br />grad_step = 000298, loss = 0.001464
<br />grad_step = 000299, loss = 0.001458
<br />grad_step = 000300, loss = 0.001461
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001465
<br />grad_step = 000302, loss = 0.001463
<br />grad_step = 000303, loss = 0.001456
<br />grad_step = 000304, loss = 0.001451
<br />grad_step = 000305, loss = 0.001451
<br />grad_step = 000306, loss = 0.001454
<br />grad_step = 000307, loss = 0.001453
<br />grad_step = 000308, loss = 0.001449
<br />grad_step = 000309, loss = 0.001445
<br />grad_step = 000310, loss = 0.001445
<br />grad_step = 000311, loss = 0.001447
<br />grad_step = 000312, loss = 0.001450
<br />grad_step = 000313, loss = 0.001453
<br />grad_step = 000314, loss = 0.001463
<br />grad_step = 000315, loss = 0.001478
<br />grad_step = 000316, loss = 0.001518
<br />grad_step = 000317, loss = 0.001539
<br />grad_step = 000318, loss = 0.001576
<br />grad_step = 000319, loss = 0.001540
<br />grad_step = 000320, loss = 0.001494
<br />grad_step = 000321, loss = 0.001450
<br />grad_step = 000322, loss = 0.001442
<br />grad_step = 000323, loss = 0.001465
<br />grad_step = 000324, loss = 0.001489
<br />grad_step = 000325, loss = 0.001496
<br />grad_step = 000326, loss = 0.001467
<br />grad_step = 000327, loss = 0.001439
<br />grad_step = 000328, loss = 0.001430
<br />grad_step = 000329, loss = 0.001436
<br />grad_step = 000330, loss = 0.001437
<br />grad_step = 000331, loss = 0.001427
<br />grad_step = 000332, loss = 0.001419
<br />grad_step = 000333, loss = 0.001425
<br />grad_step = 000334, loss = 0.001441
<br />grad_step = 000335, loss = 0.001453
<br />grad_step = 000336, loss = 0.001466
<br />grad_step = 000337, loss = 0.001468
<br />grad_step = 000338, loss = 0.001485
<br />grad_step = 000339, loss = 0.001495
<br />grad_step = 000340, loss = 0.001508
<br />grad_step = 000341, loss = 0.001481
<br />grad_step = 000342, loss = 0.001442
<br />grad_step = 000343, loss = 0.001409
<br />grad_step = 000344, loss = 0.001406
<br />grad_step = 000345, loss = 0.001420
<br />grad_step = 000346, loss = 0.001430
<br />grad_step = 000347, loss = 0.001432
<br />grad_step = 000348, loss = 0.001425
<br />grad_step = 000349, loss = 0.001419
<br />grad_step = 000350, loss = 0.001405
<br />grad_step = 000351, loss = 0.001393
<br />grad_step = 000352, loss = 0.001389
<br />grad_step = 000353, loss = 0.001396
<br />grad_step = 000354, loss = 0.001406
<br />grad_step = 000355, loss = 0.001409
<br />grad_step = 000356, loss = 0.001410
<br />grad_step = 000357, loss = 0.001407
<br />grad_step = 000358, loss = 0.001410
<br />grad_step = 000359, loss = 0.001408
<br />grad_step = 000360, loss = 0.001405
<br />grad_step = 000361, loss = 0.001395
<br />grad_step = 000362, loss = 0.001386
<br />grad_step = 000363, loss = 0.001380
<br />grad_step = 000364, loss = 0.001377
<br />grad_step = 000365, loss = 0.001374
<br />grad_step = 000366, loss = 0.001370
<br />grad_step = 000367, loss = 0.001367
<br />grad_step = 000368, loss = 0.001366
<br />grad_step = 000369, loss = 0.001367
<br />grad_step = 000370, loss = 0.001369
<br />grad_step = 000371, loss = 0.001372
<br />grad_step = 000372, loss = 0.001377
<br />grad_step = 000373, loss = 0.001389
<br />grad_step = 000374, loss = 0.001411
<br />grad_step = 000375, loss = 0.001457
<br />grad_step = 000376, loss = 0.001512
<br />grad_step = 000377, loss = 0.001601
<br />grad_step = 000378, loss = 0.001617
<br />grad_step = 000379, loss = 0.001606
<br />grad_step = 000380, loss = 0.001473
<br />grad_step = 000381, loss = 0.001371
<br />grad_step = 000382, loss = 0.001363
<br />grad_step = 000383, loss = 0.001432
<br />grad_step = 000384, loss = 0.001488
<br />grad_step = 000385, loss = 0.001446
<br />grad_step = 000386, loss = 0.001383
<br />grad_step = 000387, loss = 0.001343
<br />grad_step = 000388, loss = 0.001358
<br />grad_step = 000389, loss = 0.001403
<br />grad_step = 000390, loss = 0.001423
<br />grad_step = 000391, loss = 0.001426
<br />grad_step = 000392, loss = 0.001386
<br />grad_step = 000393, loss = 0.001350
<br />grad_step = 000394, loss = 0.001333
<br />grad_step = 000395, loss = 0.001340
<br />grad_step = 000396, loss = 0.001362
<br />grad_step = 000397, loss = 0.001373
<br />grad_step = 000398, loss = 0.001370
<br />grad_step = 000399, loss = 0.001349
<br />grad_step = 000400, loss = 0.001329
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001320
<br />grad_step = 000402, loss = 0.001323
<br />grad_step = 000403, loss = 0.001331
<br />grad_step = 000404, loss = 0.001339
<br />grad_step = 000405, loss = 0.001344
<br />grad_step = 000406, loss = 0.001343
<br />grad_step = 000407, loss = 0.001339
<br />grad_step = 000408, loss = 0.001331
<br />grad_step = 000409, loss = 0.001322
<br />grad_step = 000410, loss = 0.001313
<br />grad_step = 000411, loss = 0.001306
<br />grad_step = 000412, loss = 0.001303
<br />grad_step = 000413, loss = 0.001302
<br />grad_step = 000414, loss = 0.001303
<br />grad_step = 000415, loss = 0.001304
<br />grad_step = 000416, loss = 0.001308
<br />grad_step = 000417, loss = 0.001313
<br />grad_step = 000418, loss = 0.001322
<br />grad_step = 000419, loss = 0.001334
<br />grad_step = 000420, loss = 0.001356
<br />grad_step = 000421, loss = 0.001381
<br />grad_step = 000422, loss = 0.001424
<br />grad_step = 000423, loss = 0.001454
<br />grad_step = 000424, loss = 0.001490
<br />grad_step = 000425, loss = 0.001460
<br />grad_step = 000426, loss = 0.001405
<br />grad_step = 000427, loss = 0.001324
<br />grad_step = 000428, loss = 0.001283
<br />grad_step = 000429, loss = 0.001300
<br />grad_step = 000430, loss = 0.001344
<br />grad_step = 000431, loss = 0.001378
<br />grad_step = 000432, loss = 0.001366
<br />grad_step = 000433, loss = 0.001337
<br />grad_step = 000434, loss = 0.001297
<br />grad_step = 000435, loss = 0.001274
<br />grad_step = 000436, loss = 0.001272
<br />grad_step = 000437, loss = 0.001286
<br />grad_step = 000438, loss = 0.001308
<br />grad_step = 000439, loss = 0.001323
<br />grad_step = 000440, loss = 0.001333
<br />grad_step = 000441, loss = 0.001330
<br />grad_step = 000442, loss = 0.001321
<br />grad_step = 000443, loss = 0.001304
<br />grad_step = 000444, loss = 0.001289
<br />grad_step = 000445, loss = 0.001271
<br />grad_step = 000446, loss = 0.001259
<br />grad_step = 000447, loss = 0.001252
<br />grad_step = 000448, loss = 0.001251
<br />grad_step = 000449, loss = 0.001255
<br />grad_step = 000450, loss = 0.001262
<br />grad_step = 000451, loss = 0.001272
<br />grad_step = 000452, loss = 0.001283
<br />grad_step = 000453, loss = 0.001304
<br />grad_step = 000454, loss = 0.001328
<br />grad_step = 000455, loss = 0.001370
<br />grad_step = 000456, loss = 0.001404
<br />grad_step = 000457, loss = 0.001442
<br />grad_step = 000458, loss = 0.001423
<br />grad_step = 000459, loss = 0.001379
<br />grad_step = 000460, loss = 0.001297
<br />grad_step = 000461, loss = 0.001246
<br />grad_step = 000462, loss = 0.001242
<br />grad_step = 000463, loss = 0.001275
<br />grad_step = 000464, loss = 0.001314
<br />grad_step = 000465, loss = 0.001323
<br />grad_step = 000466, loss = 0.001309
<br />grad_step = 000467, loss = 0.001268
<br />grad_step = 000468, loss = 0.001235
<br />grad_step = 000469, loss = 0.001221
<br />grad_step = 000470, loss = 0.001225
<br />grad_step = 000471, loss = 0.001241
<br />grad_step = 000472, loss = 0.001260
<br />grad_step = 000473, loss = 0.001283
<br />grad_step = 000474, loss = 0.001299
<br />grad_step = 000475, loss = 0.001316
<br />grad_step = 000476, loss = 0.001316
<br />grad_step = 000477, loss = 0.001306
<br />grad_step = 000478, loss = 0.001275
<br />grad_step = 000479, loss = 0.001240
<br />grad_step = 000480, loss = 0.001213
<br />grad_step = 000481, loss = 0.001204
<br />grad_step = 000482, loss = 0.001212
<br />grad_step = 000483, loss = 0.001229
<br />grad_step = 000484, loss = 0.001245
<br />grad_step = 000485, loss = 0.001252
<br />grad_step = 000486, loss = 0.001256
<br />grad_step = 000487, loss = 0.001246
<br />grad_step = 000488, loss = 0.001238
<br />grad_step = 000489, loss = 0.001221
<br />grad_step = 000490, loss = 0.001209
<br />grad_step = 000491, loss = 0.001197
<br />grad_step = 000492, loss = 0.001189
<br />grad_step = 000493, loss = 0.001184
<br />grad_step = 000494, loss = 0.001183
<br />grad_step = 000495, loss = 0.001184
<br />grad_step = 000496, loss = 0.001186
<br />grad_step = 000497, loss = 0.001191
<br />grad_step = 000498, loss = 0.001200
<br />grad_step = 000499, loss = 0.001218
<br />grad_step = 000500, loss = 0.001249
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001307
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
<br />  date_run                              2020-05-24 23:57:16.974005
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   0.24937
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:57:16.980524
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   0.16186
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:57:16.987277
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   0.13814
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:57:16.992471
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -1.45953
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
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fad536bac88> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-24 23:57:35.559578
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:57:35.564376
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:57:35.568922
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-24 23:57:35.572616
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
<br />0   2020-05-24 23:56:53.492233  ...    mean_absolute_error
<br />1   2020-05-24 23:56:53.499298  ...     mean_squared_error
<br />2   2020-05-24 23:56:53.504516  ...  median_absolute_error
<br />3   2020-05-24 23:56:53.509338  ...               r2_score
<br />4   2020-05-24 23:57:16.974005  ...    mean_absolute_error
<br />5   2020-05-24 23:57:16.980524  ...     mean_squared_error
<br />6   2020-05-24 23:57:16.987277  ...  median_absolute_error
<br />7   2020-05-24 23:57:16.992471  ...               r2_score
<br />8   2020-05-24 23:57:35.559578  ...    mean_absolute_error
<br />9   2020-05-24 23:57:35.564376  ...     mean_squared_error
<br />10  2020-05-24 23:57:35.568922  ...  median_absolute_error
<br />11  2020-05-24 23:57:35.572616  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 10, [Traceback at line 3084](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3084)<br />3084..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'

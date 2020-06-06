## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 1150](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1150)<br />1150..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 2, [Traceback at line 1162](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1162)<br />1162..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 1169](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1169)<br />1169..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 4, [Traceback at line 1175](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1175)<br />1175..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 5, [Traceback at line 1187](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1187)<br />1187..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 1194](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1194)<br />1194..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 7, [Traceback at line 1228](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1228)<br />1228..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 97, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :
<br />KeyError: 'distr_output'



### Error 8, [Traceback at line 1251](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1251)<br />1251..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 9, [Traceback at line 1279](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1279)<br />1279..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 10, [Traceback at line 1306](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1306)<br />1306..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 11, [Traceback at line 1333](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1333)<br />1333..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 12, [Traceback at line 1360](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1360)<br />1360..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 13, [Traceback at line 1401](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1401)<br />1401..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 14, [Traceback at line 1411](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1411)<br />1411..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/model_gluon/gluonts_model.py", line 90, in __init__
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
<br /> ********************************************************************************************************************************************
<br />ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
<br />
<br />  dataset/json/benchmark.json 
<br />
<br />  Custom benchmark 
<br />
<br />  ['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'] 
<br />
<br />  json_path https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/ztest/model_keras/armdn/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/ztest/model_keras/armdn/'}} Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/ztest/model_fb/fb_prophet/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 



### Error 15, [Traceback at line 1478](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1478)<br />1478..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 16, [Traceback at line 1490](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1490)<br />1490..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 17, [Traceback at line 1497](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1497)<br />1497..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 18, [Traceback at line 1503](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1503)<br />1503..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 19, [Traceback at line 1515](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1515)<br />1515..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 20, [Traceback at line 1522](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1522)<br />1522..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 21, [Traceback at line 1674](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1674)<br />1674..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/af245b65de5ba5fde0bd6c86165709e91e088960/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'

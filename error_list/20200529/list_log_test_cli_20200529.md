## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 651](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L651)<br />651..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 2, [Traceback at line 1017](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1017)<br />1017..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 3, [Traceback at line 1029](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1029)<br />1029..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 4, [Traceback at line 1036](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1036)<br />1036..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 5, [Traceback at line 1042](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1042)<br />1042..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 6, [Traceback at line 1054](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1054)<br />1054..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 1061](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1061)<br />1061..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 8, [Traceback at line 1095](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1095)<br />1095..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 95, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :
<br />KeyError: 'distr_output'



### Error 9, [Traceback at line 1119](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1119)<br />1119..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 10, [Traceback at line 1148](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1148)<br />1148..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 11, [Traceback at line 1176](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1176)<br />1176..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 12, [Traceback at line 1204](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1204)<br />1204..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 13, [Traceback at line 1232](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1232)<br />1232..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 14, [Traceback at line 1274](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1274)<br />1274..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 15, [Traceback at line 1284](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1284)<br />1284..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/model_gluon/gluonts_model.py", line 90, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/ztest/model_keras/armdn/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/ztest/model_keras/armdn/'}} Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/ztest/model_fb/fb_prophet/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 



### Error 16, [Traceback at line 1351](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1351)<br />1351..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 17, [Traceback at line 1363](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1363)<br />1363..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 18, [Traceback at line 1370](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1370)<br />1370..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 19, [Traceback at line 1376](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1376)<br />1376..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 20, [Traceback at line 1388](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1388)<br />1388..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 21, [Traceback at line 1395](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1395)<br />1395..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 22, [Traceback at line 1547](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1547)<br />1547..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/64484851bea8f94cdaa6013748aa63d15238f3b7/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'

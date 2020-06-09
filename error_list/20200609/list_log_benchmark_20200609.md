## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py


### Error 1, [Traceback at line 238](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L238)<br />238..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 2, [Traceback at line 250](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L250)<br />250..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 257](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L257)<br />257..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 4, [Traceback at line 263](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L263)<br />263..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 5, [Traceback at line 275](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L275)<br />275..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 282](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L282)<br />282..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 7, [Traceback at line 288](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L288)<br />288..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 8, [Traceback at line 308](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L308)<br />308..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 315](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L315)<br />315..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 10, [Traceback at line 321](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L321)<br />321..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 11, [Traceback at line 341](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L341)<br />341..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 12, [Traceback at line 348](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L348)<br />348..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 13, [Traceback at line 354](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L354)<br />354..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 14, [Traceback at line 374](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L374)<br />374..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />
<br />  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />IndexError: tuple index out of range



### Error 15, [Traceback at line 433](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L433)<br />433..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 16, [Traceback at line 439](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L439)<br />439..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 17, [Traceback at line 459](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L459)<br />459..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 18, [Traceback at line 466](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L466)<br />466..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 19, [Traceback at line 472](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L472)<br />472..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 20, [Traceback at line 492](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L492)<br />492..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 21, [Traceback at line 499](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L499)<br />499..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 22, [Traceback at line 505](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L505)<br />505..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 23, [Traceback at line 525](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L525)<br />525..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 24, [Traceback at line 532](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L532)<br />532..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 25, [Traceback at line 538](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L538)<br />538..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 26, [Traceback at line 558](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L558)<br />558..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 27, [Traceback at line 565](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L565)<br />565..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />
<br />  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/example/benchmark/timeseries/test02/model_list.json 
<br />
<br />                       date_run  ...            metric_name
<br />0  2020-06-09 04:14:56.923853  ...    mean_absolute_error
<br />1  2020-06-09 04:14:56.927603  ...     mean_squared_error
<br />2  2020-06-09 04:14:56.930756  ...  median_absolute_error
<br />3  2020-06-09 04:14:56.934098  ...               r2_score
<br />
<br />[4 rows x 6 columns] 
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 28, [Traceback at line 596](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L596)<br />596..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_gluon/gluonts_model.py", line 152
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 29, [Traceback at line 616](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L616)<br />616..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 30, [Traceback at line 623](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L623)<br />623..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 152), tuple index out of range



### Error 31, [Traceback at line 816](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L816)<br />816..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 32, [Traceback at line 827](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L827)<br />827..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 33, [Traceback at line 838](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L838)<br />838..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 34, [Traceback at line 849](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L849)<br />849..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 35, [Traceback at line 860](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L860)<br />860..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 36, [Traceback at line 871](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L871)<br />871..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 37, [Traceback at line 882](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L882)<br />882..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 38, [Traceback at line 893](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L893)<br />893..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 39, [Traceback at line 904](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L904)<br />904..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 40, [Traceback at line 915](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L915)<br />915..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 41, [Traceback at line 926](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L926)<br />926..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0ca81786a0> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} 'data_info' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/example/benchmark/cnn/mnist 
<br />
<br />  Empty DataFrame
<br />Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
<br />Index: [] 
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 42, [Traceback at line 954](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L954)<br />954..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 284, in <module>
<br />    main()
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 281, in main
<br />    raise Exception("No options")
<br />Exception: No options
<br />python https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py --do fashion_vision_mnist 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  text_classification 
<br />
<br />  json_path https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/json/benchmark_text_classification/model_list_bench01.json 
<br />
<br />  Model List [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}] 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5a98b654e0> <class 'mlmodels.model_tch.textcnn.Model'>
<br />
<br />  {'hypermodel_pars': {}, 'data_pars': {'data_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': True}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'path': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 
<br />
<br />  #### Setup Model   ############################################## 



### Error 43, [Traceback at line 1001](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1001)<br />1001..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/textcnn.py", line 361, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/textcnn.py", line 414, in get_dataset
<br />    dataset        = data_pars['data_info'].get('dataset', None)
<br />KeyError: 'data_info'



### Error 44, [Traceback at line 1055](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1055)<br />1055..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 45, [Traceback at line 1117](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1117)<br />1117..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/textcnn.py", line 361, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/textcnn.py", line 414, in get_dataset
<br />    dataset        = data_pars['data_info'].get('dataset', None)
<br />KeyError: 'data_info'



### Error 46, [Traceback at line 1125](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1125)<br />1125..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.matchzoo_models'



### Error 47, [Traceback at line 1137](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1137)<br />1137..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 48, [Traceback at line 1144](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1144)<br />1144..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.matchzoo_models notfound, No module named 'mlmodels.model_tch.matchzoo_models', tuple index out of range



### Error 49, [Traceback at line 1266](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1266)<br />1266..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 50, [Traceback at line 1274](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1274)<br />1274..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.Autokeras'



### Error 51, [Traceback at line 1286](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1286)<br />1286..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 52, [Traceback at line 1293](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1293)<br />1293..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'mlmodels.model_keras.Autokeras', tuple index out of range



### Error 53, [Traceback at line 1299](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1299)<br />1299..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.transformer_classifier'



### Error 54, [Traceback at line 1311](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1311)<br />1311..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 55, [Traceback at line 1318](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1318)<br />1318..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.transformer_classifier notfound, No module named 'mlmodels.model_tch.transformer_classifier', tuple index out of range



### Error 56, [Traceback at line 1324](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1324)<br />1324..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/transformer_sentence.py", line 135, in fit
<br />    data_readers, interal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_tch/transformer_sentence.py", line 220, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 57, [Traceback at line 1334](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1334)<br />1334..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/model_keras/namentity_crm_bilstm.py", line 182, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 58, [Traceback at line 1344](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1344)<br />1344..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.textvae'



### Error 59, [Traceback at line 1356](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1356)<br />1356..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 60, [Traceback at line 1363](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1363)<br />1363..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.textvae notfound, No module named 'mlmodels.model_keras.textvae', tuple index out of range

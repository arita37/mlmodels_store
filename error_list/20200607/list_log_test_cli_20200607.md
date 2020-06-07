## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 451](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L451)<br />451..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 140, in get_dataset
<br />    train, test = get_dataset_gluonts(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 203, in get_dataset_gluonts
<br />    train=dataset_path / "train", test=dataset_path / "test",)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/dataset/common.py", line 477, in load_datasets
<br />    meta = MetaData.parse_file(Path(metadata) / "metadata.json")
<br />  File "pydantic/main.py", line 437, in pydantic.main.BaseModel.parse_file
<br />  File "pydantic/parse.py", line 57, in pydantic.parse.load_file
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 1189, in read_bytes
<br />    with self.open(mode='rb') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 1183, in open
<br />    opener=self._opener)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 1037, in _opener
<br />    return self._accessor.open(self, flags, mode)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 387, in wrapped
<br />    return strfunc(str(pathobj), *args)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timseries/m4_dataset/m4_daily/metadata.json'



### Error 2, [Traceback at line 511](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L511)<br />511..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 140, in get_dataset
<br />    train, test = get_dataset_gluonts(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 203, in get_dataset_gluonts
<br />    train=dataset_path / "train", test=dataset_path / "test",)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/dataset/common.py", line 477, in load_datasets
<br />    meta = MetaData.parse_file(Path(metadata) / "metadata.json")
<br />  File "pydantic/main.py", line 437, in pydantic.main.BaseModel.parse_file
<br />  File "pydantic/parse.py", line 57, in pydantic.parse.load_file
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 1189, in read_bytes
<br />    with self.open(mode='rb') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 1183, in open
<br />    opener=self._opener)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 1037, in _opener
<br />    return self._accessor.open(self, flags, mode)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pathlib.py", line 387, in wrapped
<br />    return strfunc(str(pathobj), *args)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'gluonts_data/m5_dataset/metadata.json'



### Error 3, [Traceback at line 1270](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1270)<br />1270..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 4, [Traceback at line 1282](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1282)<br />1282..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 5, [Traceback at line 1289](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1289)<br />1289..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 6, [Traceback at line 1295](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1295)<br />1295..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 7, [Traceback at line 1307](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1307)<br />1307..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 8, [Traceback at line 1314](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1314)<br />1314..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 9, [Traceback at line 1348](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1348)<br />1348..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 97, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :
<br />KeyError: 'distr_output'



### Error 10, [Traceback at line 1371](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1371)<br />1371..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 11, [Traceback at line 1399](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1399)<br />1399..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 12, [Traceback at line 1426](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1426)<br />1426..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 13, [Traceback at line 1453](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1453)<br />1453..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 14, [Traceback at line 1480](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1480)<br />1480..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 15, [Traceback at line 1521](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1521)<br />1521..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 16, [Traceback at line 1531](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1531)<br />1531..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/model_gluon/gluonts_model.py", line 90, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/ztest/model_keras/armdn/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/ztest/model_keras/armdn/'}} Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/ztest/model_fb/fb_prophet/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 



### Error 17, [Traceback at line 1598](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1598)<br />1598..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 18, [Traceback at line 1610](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1610)<br />1610..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 19, [Traceback at line 1617](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1617)<br />1617..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 20, [Traceback at line 1623](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1623)<br />1623..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 21, [Traceback at line 1635](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1635)<br />1635..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 22, [Traceback at line 1642](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1642)<br />1642..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 23, [Traceback at line 1794](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1794)<br />1794..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3351645c97b9d6d9cb5bd0fae6734129deb628ce/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'

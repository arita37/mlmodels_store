## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 50](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L50)<br />50..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/textcnn.py", line 24, in <module>
<br />    import torchtext
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
<br />    _init_extension()
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
<br />    torch.ops.load_library(ext_specs.origin)
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
<br />    ctypes.CDLL(path)
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
<br />    self._handle = _dlopen(self._name, mode)
<br />OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory



### Error 2, [Traceback at line 75](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L75)<br />75..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 82](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L82)<br />82..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.textcnn notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range



### Error 4, [Traceback at line 103](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L103)<br />103..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 133](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L133)<br />133..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.textvae'



### Error 6, [Traceback at line 145](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L145)<br />145..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 152](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L152)<br />152..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.textvae notfound, No module named 'mlmodels.model_keras.textvae', tuple index out of range



### Error 8, [Traceback at line 173](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L173)<br />173..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 240](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L240)<br />240..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 10, [Traceback at line 265](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L265)<br />265..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 293](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L293)<br />293..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 12, [Traceback at line 312](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L312)<br />312..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 13, [Traceback at line 342](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L342)<br />342..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 14, [Traceback at line 359](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L359)<br />359..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 15, [Traceback at line 366](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L366)<br />366..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 16, [Traceback at line 387](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L387)<br />387..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 17, [Traceback at line 418](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L418)<br />418..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras/namentity_crm_bilstm.py", line 182, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 18, [Traceback at line 445](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L445)<br />445..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 478](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L478)<br />478..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 20, [Traceback at line 505](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L505)<br />505..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 538](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L538)<br />538..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 22, [Traceback at line 565](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L565)<br />565..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 598](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L598)<br />598..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 24, [Traceback at line 625](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L625)<br />625..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 25, [Traceback at line 658](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L658)<br />658..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 26, [Traceback at line 685](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L685)<br />685..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 718](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L718)<br />718..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 28, [Traceback at line 745](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L745)<br />745..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 29, [Traceback at line 778](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L778)<br />778..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 30, [Traceback at line 805](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L805)<br />805..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 31, [Traceback at line 838](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L838)<br />838..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 32, [Traceback at line 865](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L865)<br />865..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 33, [Traceback at line 898](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L898)<br />898..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 34, [Traceback at line 925](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L925)<br />925..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 35, [Traceback at line 958](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L958)<br />958..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 36, [Traceback at line 985](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L985)<br />985..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 37, [Traceback at line 1018](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1018)<br />1018..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 38, [Traceback at line 1045](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1045)<br />1045..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 39, [Traceback at line 1078](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1078)<br />1078..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 40, [Traceback at line 1105](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1105)<br />1105..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 41, [Traceback at line 1138](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1138)<br />1138..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 42, [Traceback at line 1165](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1165)<br />1165..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 43, [Traceback at line 1198](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1198)<br />1198..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 44, [Traceback at line 1225](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1225)<br />1225..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 45, [Traceback at line 1258](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1258)<br />1258..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 46, [Traceback at line 1285](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1285)<br />1285..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 47, [Traceback at line 1318](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1318)<br />1318..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 48, [Traceback at line 1345](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1345)<br />1345..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 49, [Traceback at line 1378](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1378)<br />1378..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 50, [Traceback at line 1405](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1405)<br />1405..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 51, [Traceback at line 1438](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1438)<br />1438..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 52, [Traceback at line 1465](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1465)<br />1465..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 53, [Traceback at line 1498](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1498)<br />1498..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 54, [Traceback at line 1525](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1525)<br />1525..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 55, [Traceback at line 1558](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1558)<br />1558..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 56, [Traceback at line 1585](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1585)<br />1585..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 57, [Traceback at line 1618](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1618)<br />1618..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 58, [Traceback at line 1645](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1645)<br />1645..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 59, [Traceback at line 1678](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1678)<br />1678..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 60, [Traceback at line 1705](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1705)<br />1705..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 61, [Traceback at line 1738](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1738)<br />1738..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 62, [Traceback at line 1765](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1765)<br />1765..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 63, [Traceback at line 1798](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1798)<br />1798..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 64, [Traceback at line 1825](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1825)<br />1825..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 65, [Traceback at line 1858](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1858)<br />1858..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 66, [Traceback at line 1885](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1885)<br />1885..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 67, [Traceback at line 1918](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1918)<br />1918..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 68, [Traceback at line 1945](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1945)<br />1945..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 69, [Traceback at line 1978](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1978)<br />1978..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 70, [Traceback at line 2005](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2005)<br />2005..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 71, [Traceback at line 2035](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2035)<br />2035..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 72, [Traceback at line 2055](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2055)<br />2055..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 73, [Traceback at line 2062](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2062)<br />2062..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 74, [Traceback at line 2085](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2085)<br />2085..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 75, [Traceback at line 2105](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2105)<br />2105..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 76, [Traceback at line 2112](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2112)<br />2112..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 77, [Traceback at line 2135](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2135)<br />2135..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 78, [Traceback at line 2155](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2155)<br />2155..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 79, [Traceback at line 2162](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2162)<br />2162..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 80, [Traceback at line 2185](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2185)<br />2185..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 81, [Traceback at line 2205](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2205)<br />2205..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 82, [Traceback at line 2212](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2212)<br />2212..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 83, [Traceback at line 2235](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2235)<br />2235..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 84, [Traceback at line 2255](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2255)<br />2255..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 85, [Traceback at line 2262](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2262)<br />2262..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 86, [Traceback at line 2285](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2285)<br />2285..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 87, [Traceback at line 2305](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2305)<br />2305..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 88, [Traceback at line 2312](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2312)<br />2312..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 89, [Traceback at line 2335](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2335)<br />2335..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 90, [Traceback at line 2355](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2355)<br />2355..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 91, [Traceback at line 2362](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2362)<br />2362..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 92, [Traceback at line 2385](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2385)<br />2385..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 93, [Traceback at line 2405](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2405)<br />2405..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 94, [Traceback at line 2412](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2412)<br />2412..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 95, [Traceback at line 2438](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2438)<br />2438..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 96, [Traceback at line 2457](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2457)<br />2457..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 97, [Traceback at line 2476](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2476)<br />2476..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 98, [Traceback at line 2497](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2497)<br />2497..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 99, [Traceback at line 2517](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2517)<br />2517..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 100, [Traceback at line 2524](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2524)<br />2524..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 101, [Traceback at line 2547](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2547)<br />2547..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 102, [Traceback at line 2567](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2567)<br />2567..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 103, [Traceback at line 2574](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2574)<br />2574..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 104, [Traceback at line 2597](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2597)<br />2597..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 105, [Traceback at line 2617](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2617)<br />2617..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 106, [Traceback at line 2624](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2624)<br />2624..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 107, [Traceback at line 2647](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2647)<br />2647..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 108, [Traceback at line 2667](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2667)<br />2667..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 109, [Traceback at line 2674](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2674)<br />2674..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 110, [Traceback at line 2697](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2697)<br />2697..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 111, [Traceback at line 2717](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2717)<br />2717..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 112, [Traceback at line 2724](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2724)<br />2724..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 113, [Traceback at line 2747](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2747)<br />2747..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 114, [Traceback at line 2767](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2767)<br />2767..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 115, [Traceback at line 2774](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2774)<br />2774..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 116, [Traceback at line 2797](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2797)<br />2797..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 117, [Traceback at line 2817](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2817)<br />2817..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 118, [Traceback at line 2824](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2824)<br />2824..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 119, [Traceback at line 2847](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2847)<br />2847..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 120, [Traceback at line 2867](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2867)<br />2867..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 121, [Traceback at line 2874](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2874)<br />2874..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 122, [Traceback at line 2900](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2900)<br />2900..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 123, [Traceback at line 2924](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2924)<br />2924..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 124, [Traceback at line 2948](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2948)<br />2948..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 125, [Traceback at line 2967](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2967)<br />2967..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 126, [Traceback at line 2995](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2995)<br />2995..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 127, [Traceback at line 3019](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3019)<br />3019..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 128, [Traceback at line 3043](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3043)<br />3043..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 129, [Traceback at line 3067](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3067)<br />3067..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 130, [Traceback at line 3086](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3086)<br />3086..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 131, [Traceback at line 3114](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3114)<br />3114..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 132, [Traceback at line 3133](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3133)<br />3133..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 133, [Traceback at line 3161](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3161)<br />3161..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 134, [Traceback at line 3185](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3185)<br />3185..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 135, [Traceback at line 3204](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3204)<br />3204..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 136, [Traceback at line 3232](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3232)<br />3232..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 137, [Traceback at line 3256](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3256)<br />3256..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 138, [Traceback at line 3280](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3280)<br />3280..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 139, [Traceback at line 3304](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3304)<br />3304..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 140, [Traceback at line 3323](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3323)<br />3323..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 141, [Traceback at line 3351](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3351)<br />3351..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 142, [Traceback at line 3370](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3370)<br />3370..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

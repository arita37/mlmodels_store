## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 53](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L53)<br />53..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/textcnn.py", line 361, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/textcnn.py", line 414, in get_dataset
<br />    dataset        = data_pars['data_info'].get('dataset', None)
<br />KeyError: 'data_info'



### Error 2, [Traceback at line 78](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L78)<br />78..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 3, [Traceback at line 145](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L145)<br />145..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 4, [Traceback at line 170](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L170)<br />170..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 200](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L200)<br />200..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.Autokeras'



### Error 6, [Traceback at line 212](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L212)<br />212..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 219](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L219)<br />219..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'mlmodels.model_keras.Autokeras', tuple index out of range



### Error 8, [Traceback at line 240](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L240)<br />240..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 268](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L268)<br />268..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 287](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L287)<br />287..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 318](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L318)<br />318..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_keras/namentity_crm_bilstm.py", line 182, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 12, [Traceback at line 345](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L345)<br />345..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 13, [Traceback at line 375](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L375)<br />375..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.textvae'



### Error 14, [Traceback at line 387](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L387)<br />387..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 15, [Traceback at line 394](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L394)<br />394..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.textvae notfound, No module named 'mlmodels.model_keras.textvae', tuple index out of range



### Error 16, [Traceback at line 415](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L415)<br />415..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 17, [Traceback at line 443](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L443)<br />443..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 462](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L462)<br />462..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 490](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L490)<br />490..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 514](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L514)<br />514..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 21, [Traceback at line 533](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L533)<br />533..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 22, [Traceback at line 561](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L561)<br />561..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 23, [Traceback at line 580](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L580)<br />580..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 24, [Traceback at line 608](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L608)<br />608..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 25, [Traceback at line 627](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L627)<br />627..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 26, [Traceback at line 655](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L655)<br />655..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 27, [Traceback at line 679](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L679)<br />679..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 28, [Traceback at line 703](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L703)<br />703..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 722](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L722)<br />722..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 30, [Traceback at line 750](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L750)<br />750..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 31, [Traceback at line 774](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L774)<br />774..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 32, [Traceback at line 798](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L798)<br />798..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 33, [Traceback at line 822](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L822)<br />822..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 34, [Traceback at line 841](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L841)<br />841..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 35, [Traceback at line 860](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L860)<br />860..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 36, [Traceback at line 896](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L896)<br />896..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 92, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :  
<br />KeyError: 'distr_output'



### Error 37, [Traceback at line 940](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L940)<br />940..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 38, [Traceback at line 989](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L989)<br />989..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 39, [Traceback at line 1037](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1037)<br />1037..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 40, [Traceback at line 1085](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1085)<br />1085..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 41, [Traceback at line 1133](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1133)<br />1133..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 42, [Traceback at line 1181](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1181)<br />1181..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 43, [Traceback at line 1225](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1225)<br />1225..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 86, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json  --config_mode deepar  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />  ##### Init model_gluon.gluonts_model {'path': 'https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/ztest/model_gluon/gluonts_deepar/', 'model_uri': 'model_gluon.gluonts_model'} 
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU
<br />INFO:root:Using CPU



### Error 44, [Traceback at line 1275](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1275)<br />1275..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 92, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :  
<br />KeyError: 'distr_output'



### Error 45, [Traceback at line 1319](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1319)<br />1319..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 46, [Traceback at line 1368](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1368)<br />1368..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 47, [Traceback at line 1416](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1416)<br />1416..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 48, [Traceback at line 1464](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1464)<br />1464..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 49, [Traceback at line 1512](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1512)<br />1512..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 50, [Traceback at line 1560](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1560)<br />1560..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 256, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 134, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 210, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 51, [Traceback at line 1604](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1604)<br />1604..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_gluon/gluonts_model.py", line 86, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json  --config_mode test  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 



### Error 52, [Traceback at line 1637](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1637)<br />1637..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 53, [Traceback at line 1661](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1661)<br />1661..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 54, [Traceback at line 1680](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1680)<br />1680..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 55, [Traceback at line 1708](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1708)<br />1708..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 56, [Traceback at line 1732](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1732)<br />1732..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 57, [Traceback at line 1761](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1761)<br />1761..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 58, [Traceback at line 1788](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1788)<br />1788..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 59, [Traceback at line 1821](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1821)<br />1821..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 60, [Traceback at line 1848](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1848)<br />1848..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 61, [Traceback at line 1881](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1881)<br />1881..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 62, [Traceback at line 1908](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1908)<br />1908..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 63, [Traceback at line 1941](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1941)<br />1941..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 64, [Traceback at line 1968](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1968)<br />1968..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 65, [Traceback at line 2001](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2001)<br />2001..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 66, [Traceback at line 2028](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2028)<br />2028..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 67, [Traceback at line 2061](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2061)<br />2061..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 68, [Traceback at line 2088](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2088)<br />2088..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 69, [Traceback at line 2121](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2121)<br />2121..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 70, [Traceback at line 2148](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2148)<br />2148..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 71, [Traceback at line 2181](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2181)<br />2181..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 72, [Traceback at line 2208](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2208)<br />2208..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 73, [Traceback at line 2241](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2241)<br />2241..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 74, [Traceback at line 2268](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2268)<br />2268..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 75, [Traceback at line 2301](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2301)<br />2301..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 76, [Traceback at line 2328](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2328)<br />2328..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 77, [Traceback at line 2361](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2361)<br />2361..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 78, [Traceback at line 2388](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2388)<br />2388..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 79, [Traceback at line 2421](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2421)<br />2421..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 80, [Traceback at line 2448](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2448)<br />2448..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 81, [Traceback at line 2481](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2481)<br />2481..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 82, [Traceback at line 2508](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2508)<br />2508..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 83, [Traceback at line 2541](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2541)<br />2541..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 84, [Traceback at line 2568](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2568)<br />2568..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 85, [Traceback at line 2601](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2601)<br />2601..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 86, [Traceback at line 2628](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2628)<br />2628..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 87, [Traceback at line 2661](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2661)<br />2661..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 88, [Traceback at line 2688](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2688)<br />2688..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 89, [Traceback at line 2721](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2721)<br />2721..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 90, [Traceback at line 2748](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2748)<br />2748..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 91, [Traceback at line 2781](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2781)<br />2781..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 92, [Traceback at line 2808](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2808)<br />2808..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 93, [Traceback at line 2841](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2841)<br />2841..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 94, [Traceback at line 2868](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2868)<br />2868..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 95, [Traceback at line 2901](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2901)<br />2901..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 96, [Traceback at line 2928](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2928)<br />2928..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 97, [Traceback at line 2961](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2961)<br />2961..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 98, [Traceback at line 2988](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2988)<br />2988..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 99, [Traceback at line 3021](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3021)<br />3021..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 100, [Traceback at line 3048](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3048)<br />3048..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 101, [Traceback at line 3081](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3081)<br />3081..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 102, [Traceback at line 3108](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3108)<br />3108..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 103, [Traceback at line 3141](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3141)<br />3141..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 104, [Traceback at line 3168](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3168)<br />3168..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 105, [Traceback at line 3201](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3201)<br />3201..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 106, [Traceback at line 3228](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3228)<br />3228..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 107, [Traceback at line 3261](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3261)<br />3261..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 108, [Traceback at line 3288](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3288)<br />3288..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/21d1b6dd356ca57a695b113d5df2de1628b16e43/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

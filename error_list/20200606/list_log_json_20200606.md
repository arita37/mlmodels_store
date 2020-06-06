## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 53](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L53)<br />53..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/textcnn.py", line 361, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/textcnn.py", line 414, in get_dataset
<br />    dataset        = data_pars['data_info'].get('dataset', None)
<br />KeyError: 'data_info'



### Error 2, [Traceback at line 78](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L78)<br />78..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 3, [Traceback at line 145](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L145)<br />145..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 4, [Traceback at line 170](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L170)<br />170..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 200](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L200)<br />200..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.Autokeras'



### Error 6, [Traceback at line 212](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L212)<br />212..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 219](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L219)<br />219..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'mlmodels.model_keras.Autokeras', tuple index out of range



### Error 8, [Traceback at line 240](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L240)<br />240..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 268](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L268)<br />268..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 287](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L287)<br />287..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 318](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L318)<br />318..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_keras/namentity_crm_bilstm.py", line 182, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 12, [Traceback at line 345](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L345)<br />345..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 13, [Traceback at line 375](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L375)<br />375..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.textvae'



### Error 14, [Traceback at line 387](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L387)<br />387..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 15, [Traceback at line 394](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L394)<br />394..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.textvae notfound, No module named 'mlmodels.model_keras.textvae', tuple index out of range



### Error 16, [Traceback at line 415](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L415)<br />415..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 17, [Traceback at line 443](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L443)<br />443..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 462](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L462)<br />462..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 490](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L490)<br />490..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 514](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L514)<br />514..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 21, [Traceback at line 533](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L533)<br />533..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 22, [Traceback at line 561](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L561)<br />561..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 23, [Traceback at line 580](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L580)<br />580..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 24, [Traceback at line 608](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L608)<br />608..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 25, [Traceback at line 627](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L627)<br />627..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 26, [Traceback at line 655](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L655)<br />655..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 27, [Traceback at line 679](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L679)<br />679..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 28, [Traceback at line 703](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L703)<br />703..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 722](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L722)<br />722..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 30, [Traceback at line 750](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L750)<br />750..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 31, [Traceback at line 774](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L774)<br />774..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 32, [Traceback at line 798](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L798)<br />798..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 33, [Traceback at line 822](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L822)<br />822..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 34, [Traceback at line 841](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L841)<br />841..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 35, [Traceback at line 860](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L860)<br />860..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 36, [Traceback at line 896](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L896)<br />896..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 97, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :  
<br />KeyError: 'distr_output'



### Error 37, [Traceback at line 939](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L939)<br />939..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 38, [Traceback at line 987](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L987)<br />987..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 39, [Traceback at line 1034](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1034)<br />1034..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 40, [Traceback at line 1081](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1081)<br />1081..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 41, [Traceback at line 1128](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1128)<br />1128..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 42, [Traceback at line 1175](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1175)<br />1175..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 43, [Traceback at line 1219](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1219)<br />1219..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 90, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json  --config_mode deepar  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />  ##### Init model_gluon.gluonts_model {'path': 'https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/ztest/model_gluon/gluonts_deepar/', 'model_uri': 'model_gluon.gluonts_model'} 
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



### Error 44, [Traceback at line 1269](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1269)<br />1269..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 97, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :  
<br />KeyError: 'distr_output'



### Error 45, [Traceback at line 1312](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1312)<br />1312..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 46, [Traceback at line 1360](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1360)<br />1360..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 47, [Traceback at line 1407](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1407)<br />1407..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 48, [Traceback at line 1454](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1454)<br />1454..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 49, [Traceback at line 1501](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1501)<br />1501..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 50, [Traceback at line 1548](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1548)<br />1548..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 278, in fit
<br />    train_ds, test_ds  = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 143, in get_dataset
<br />    train, test = get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 221, in get_dataset_single
<br />    data_path = data_pars['data_path']
<br />KeyError: 'data_path'



### Error 51, [Traceback at line 1592](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1592)<br />1592..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_gluon/gluonts_model.py", line 90, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json  --config_mode test  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 



### Error 52, [Traceback at line 1625](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1625)<br />1625..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 53, [Traceback at line 1649](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1649)<br />1649..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 54, [Traceback at line 1668](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1668)<br />1668..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 55, [Traceback at line 1696](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1696)<br />1696..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 56, [Traceback at line 1720](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1720)<br />1720..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 57, [Traceback at line 1749](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1749)<br />1749..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 58, [Traceback at line 1776](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1776)<br />1776..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 59, [Traceback at line 1809](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1809)<br />1809..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 60, [Traceback at line 1836](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1836)<br />1836..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 61, [Traceback at line 1869](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1869)<br />1869..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 62, [Traceback at line 1896](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1896)<br />1896..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 63, [Traceback at line 1929](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1929)<br />1929..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 64, [Traceback at line 1956](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1956)<br />1956..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 65, [Traceback at line 1989](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1989)<br />1989..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 66, [Traceback at line 2016](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2016)<br />2016..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 67, [Traceback at line 2049](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2049)<br />2049..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 68, [Traceback at line 2076](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2076)<br />2076..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 69, [Traceback at line 2109](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2109)<br />2109..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 70, [Traceback at line 2136](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2136)<br />2136..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 71, [Traceback at line 2169](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2169)<br />2169..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 72, [Traceback at line 2196](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2196)<br />2196..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 73, [Traceback at line 2229](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2229)<br />2229..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 74, [Traceback at line 2256](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2256)<br />2256..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 75, [Traceback at line 2289](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2289)<br />2289..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 76, [Traceback at line 2316](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2316)<br />2316..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 77, [Traceback at line 2349](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2349)<br />2349..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 78, [Traceback at line 2376](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2376)<br />2376..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 79, [Traceback at line 2409](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2409)<br />2409..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 80, [Traceback at line 2436](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2436)<br />2436..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 81, [Traceback at line 2469](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2469)<br />2469..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 82, [Traceback at line 2496](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2496)<br />2496..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 83, [Traceback at line 2529](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2529)<br />2529..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 84, [Traceback at line 2556](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2556)<br />2556..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 85, [Traceback at line 2589](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2589)<br />2589..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 86, [Traceback at line 2616](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2616)<br />2616..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 87, [Traceback at line 2649](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2649)<br />2649..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 88, [Traceback at line 2676](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2676)<br />2676..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 89, [Traceback at line 2709](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2709)<br />2709..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 90, [Traceback at line 2736](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2736)<br />2736..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 91, [Traceback at line 2769](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2769)<br />2769..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 92, [Traceback at line 2796](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2796)<br />2796..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 93, [Traceback at line 2829](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2829)<br />2829..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 94, [Traceback at line 2856](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2856)<br />2856..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 95, [Traceback at line 2889](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2889)<br />2889..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 96, [Traceback at line 2916](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2916)<br />2916..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 97, [Traceback at line 2949](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2949)<br />2949..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 98, [Traceback at line 2976](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2976)<br />2976..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 99, [Traceback at line 3009](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3009)<br />3009..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 100, [Traceback at line 3036](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3036)<br />3036..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 101, [Traceback at line 3069](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3069)<br />3069..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 102, [Traceback at line 3096](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3096)<br />3096..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 103, [Traceback at line 3129](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3129)<br />3129..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 104, [Traceback at line 3156](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3156)<br />3156..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 105, [Traceback at line 3189](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3189)<br />3189..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 106, [Traceback at line 3216](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3216)<br />3216..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 107, [Traceback at line 3249](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3249)<br />3249..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/dataloader.py", line 209, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 108, [Traceback at line 3276](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3276)<br />3276..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

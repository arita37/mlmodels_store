## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py


### Error 1, [Traceback at line 91](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L91)<br />91..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 102](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L102)<br />102..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 133](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L133)<br />133..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 1039](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1039)<br />1039..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 1069](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1069)<br />1069..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 1086](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1086)<br />1086..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 1093](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1093)<br />1093..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 1114](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1114)<br />1114..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 1142](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1142)<br />1142..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 1161](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1161)<br />1161..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 1242](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1242)<br />1242..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 1273](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1273)<br />1273..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 418, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 1300](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1300)<br />1300..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 1328](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1328)<br />1328..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 1347](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1347)<br />1347..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 1375](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1375)<br />1375..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 1399](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1399)<br />1399..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 1418](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1418)<br />1418..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 1446](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1446)<br />1446..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 1465](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1465)<br />1465..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 1493](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1493)<br />1493..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 1512](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1512)<br />1512..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 1540](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1540)<br />1540..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 1564](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1564)<br />1564..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 1588](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1588)<br />1588..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 1607](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1607)<br />1607..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 1635](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1635)<br />1635..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 1659](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1659)<br />1659..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 1683](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1683)<br />1683..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 1707](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1707)<br />1707..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 1726](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1726)<br />1726..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 1745](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1745)<br />1745..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 1766](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1766)<br />1766..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 34, [Traceback at line 1796](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1796)<br />1796..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 35, [Traceback at line 1803](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1803)<br />1803..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 36, [Traceback at line 1826](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1826)<br />1826..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 37, [Traceback at line 1856](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1856)<br />1856..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 38, [Traceback at line 1863](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1863)<br />1863..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 39, [Traceback at line 1886](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1886)<br />1886..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 40, [Traceback at line 1916](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1916)<br />1916..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 41, [Traceback at line 1923](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1923)<br />1923..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 42, [Traceback at line 1946](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1946)<br />1946..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 43, [Traceback at line 1976](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1976)<br />1976..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 44, [Traceback at line 1983](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L1983)<br />1983..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 45, [Traceback at line 2006](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2006)<br />2006..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 46, [Traceback at line 2036](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2036)<br />2036..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 47, [Traceback at line 2043](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2043)<br />2043..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 48, [Traceback at line 2066](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2066)<br />2066..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 49, [Traceback at line 2096](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2096)<br />2096..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 50, [Traceback at line 2103](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2103)<br />2103..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 51, [Traceback at line 2126](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2126)<br />2126..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 52, [Traceback at line 2156](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2156)<br />2156..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 53, [Traceback at line 2163](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2163)<br />2163..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 54, [Traceback at line 2186](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2186)<br />2186..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 55, [Traceback at line 2216](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2216)<br />2216..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 56, [Traceback at line 2223](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2223)<br />2223..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 57, [Traceback at line 2251](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2251)<br />2251..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 58, [Traceback at line 2281](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2281)<br />2281..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 59, [Traceback at line 2288](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2288)<br />2288..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 60, [Traceback at line 2311](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2311)<br />2311..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 61, [Traceback at line 2341](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2341)<br />2341..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 62, [Traceback at line 2348](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2348)<br />2348..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 63, [Traceback at line 2371](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2371)<br />2371..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 64, [Traceback at line 2401](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2401)<br />2401..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 65, [Traceback at line 2408](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2408)<br />2408..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 66, [Traceback at line 2431](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2431)<br />2431..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 67, [Traceback at line 2461](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2461)<br />2461..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 68, [Traceback at line 2468](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2468)<br />2468..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 69, [Traceback at line 2491](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2491)<br />2491..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 70, [Traceback at line 2521](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2521)<br />2521..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 71, [Traceback at line 2528](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2528)<br />2528..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 72, [Traceback at line 2551](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2551)<br />2551..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 73, [Traceback at line 2581](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2581)<br />2581..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 74, [Traceback at line 2588](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2588)<br />2588..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 75, [Traceback at line 2611](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2611)<br />2611..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 76, [Traceback at line 2641](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2641)<br />2641..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 77, [Traceback at line 2648](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2648)<br />2648..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 78, [Traceback at line 2671](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2671)<br />2671..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 79, [Traceback at line 2701](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2701)<br />2701..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 80, [Traceback at line 2708](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2708)<br />2708..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 81, [Traceback at line 2734](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2734)<br />2734..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 82, [Traceback at line 2758](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2758)<br />2758..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 83, [Traceback at line 2777](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2777)<br />2777..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2805](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2805)<br />2805..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 85, [Traceback at line 2829](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2829)<br />2829..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 86, [Traceback at line 2888](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2888)<br />2888..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 87, [Traceback at line 2923](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2923)<br />2923..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 2961](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2961)<br />2961..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 89, [Traceback at line 2996](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L2996)<br />2996..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 3034](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3034)<br />3034..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 91, [Traceback at line 3069](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3069)<br />3069..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 3107](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3107)<br />3107..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 93, [Traceback at line 3142](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3142)<br />3142..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 3180](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3180)<br />3180..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 95, [Traceback at line 3215](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3215)<br />3215..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 3253](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3253)<br />3253..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 97, [Traceback at line 3288](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3288)<br />3288..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 3326](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3326)<br />3326..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 99, [Traceback at line 3361](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3361)<br />3361..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 3399](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3399)<br />3399..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 101, [Traceback at line 3434](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3434)<br />3434..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 3472](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3472)<br />3472..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 103, [Traceback at line 3507](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3507)<br />3507..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 3545](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3545)<br />3545..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 105, [Traceback at line 3580](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3580)<br />3580..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 106, [Traceback at line 3618](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3618)<br />3618..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 107, [Traceback at line 3653](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3653)<br />3653..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 108, [Traceback at line 3691](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3691)<br />3691..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 109, [Traceback at line 3726](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3726)<br />3726..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 110, [Traceback at line 3764](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3764)<br />3764..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 111, [Traceback at line 3799](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3799)<br />3799..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 112, [Traceback at line 3837](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3837)<br />3837..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 113, [Traceback at line 3872](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3872)<br />3872..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 114, [Traceback at line 3910](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3910)<br />3910..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 115, [Traceback at line 3945](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3945)<br />3945..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 116, [Traceback at line 3983](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L3983)<br />3983..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 117, [Traceback at line 4018](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4018)<br />4018..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 118, [Traceback at line 4056](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4056)<br />4056..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 119, [Traceback at line 4091](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4091)<br />4091..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 120, [Traceback at line 4129](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4129)<br />4129..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 121, [Traceback at line 4164](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4164)<br />4164..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 122, [Traceback at line 4202](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4202)<br />4202..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 123, [Traceback at line 4237](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4237)<br />4237..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 124, [Traceback at line 4275](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4275)<br />4275..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 125, [Traceback at line 4310](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4310)<br />4310..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 126, [Traceback at line 4348](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4348)<br />4348..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 127, [Traceback at line 4383](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4383)<br />4383..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 128, [Traceback at line 4421](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4421)<br />4421..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 129, [Traceback at line 4456](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4456)<br />4456..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 130, [Traceback at line 4494](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4494)<br />4494..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 131, [Traceback at line 4529](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4529)<br />4529..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 132, [Traceback at line 4567](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4567)<br />4567..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 133, [Traceback at line 4602](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4602)<br />4602..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 134, [Traceback at line 4640](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4640)<br />4640..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 135, [Traceback at line 4675](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4675)<br />4675..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 136, [Traceback at line 4713](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4713)<br />4713..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/model_tch/torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 137, [Traceback at line 4748](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-14-16_7b5cadddfd3cd634315b570fd301533da1b0a441.py#L4748)<br />4748..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/7b5cadddfd3cd634315b570fd301533da1b0a441/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

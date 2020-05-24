## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 95](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L95)<br />95..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 106](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L106)<br />106..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 137](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L137)<br />137..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 263](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L263)<br />263..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 293](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L293)<br />293..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 310](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L310)<br />310..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 317](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L317)<br />317..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 338](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L338)<br />338..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 366](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L366)<br />366..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 385](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L385)<br />385..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 466](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L466)<br />466..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 497](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L497)<br />497..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 524](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L524)<br />524..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 552](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L552)<br />552..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 571](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L571)<br />571..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 599](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L599)<br />599..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 623](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L623)<br />623..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 642](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L642)<br />642..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 670](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L670)<br />670..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 689](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L689)<br />689..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 717](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L717)<br />717..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 736](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L736)<br />736..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 764](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L764)<br />764..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 788](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L788)<br />788..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 812](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L812)<br />812..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 831](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L831)<br />831..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 859](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L859)<br />859..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 883](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L883)<br />883..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 907](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L907)<br />907..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 931](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L931)<br />931..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 950](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L950)<br />950..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 969](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L969)<br />969..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 1023](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1023)<br />1023..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 34, [Traceback at line 1077](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1077)<br />1077..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 35, [Traceback at line 1136](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1136)<br />1136..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 36, [Traceback at line 1190](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1190)<br />1190..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 37, [Traceback at line 1254](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1254)<br />1254..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 38, [Traceback at line 1308](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1308)<br />1308..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 39, [Traceback at line 1361](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1361)<br />1361..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 40, [Traceback at line 1397](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1397)<br />1397..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json  --config_mode deepar  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />  ##### Init model_gluon.gluonts_model {'path': 'https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/ztest/model_gluon/gluonts_deepar/', 'model_uri': 'model_gluon.gluonts_model'} 
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
<br />INFO:root:Using CPU
<br />
<br />  ##### Fit <mlmodels.model_gluon.gluonts_model.Model object at 0x7f3749d62d68> 
<br />INFO:root:Start model training
<br />INFO:root:Epoch[0] Learning rate is 0.001
<br />
<br />  0%|          | 0/10 [00:00<?, ?it/s]INFO:numexpr.utils:NumExpr defaulting to 2 threads.
<br />INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
<br />
<br />100%|██████████| 10/10 [00:02<00:00,  4.66it/s, avg_epoch_loss=5.22]
<br />INFO:root:Epoch[0] Elapsed time 2.151 seconds
<br />INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.223534
<br />INFO:root:Loading parameters from best epoch (0)
<br />INFO:root:Final loss: 5.223533964157104 (occurred at epoch 0)
<br />INFO:root:End model training
<br />[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
<br />{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
<br />learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.



### Error 41, [Traceback at line 1465](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1465)<br />1465..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 42, [Traceback at line 1519](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1519)<br />1519..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 43, [Traceback at line 1578](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1578)<br />1578..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 44, [Traceback at line 1632](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1632)<br />1632..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 45, [Traceback at line 1696](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1696)<br />1696..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 46, [Traceback at line 1750](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1750)<br />1750..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 47, [Traceback at line 1803](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1803)<br />1803..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 48, [Traceback at line 1839](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1839)<br />1839..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json  --config_mode test  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 



### Error 49, [Traceback at line 1872](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1872)<br />1872..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 50, [Traceback at line 1896](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1896)<br />1896..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 51, [Traceback at line 1915](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1915)<br />1915..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 52, [Traceback at line 1943](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1943)<br />1943..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 53, [Traceback at line 1967](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1967)<br />1967..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 54, [Traceback at line 1996](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1996)<br />1996..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 55, [Traceback at line 2023](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2023)<br />2023..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 56, [Traceback at line 2056](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2056)<br />2056..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 57, [Traceback at line 2083](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2083)<br />2083..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 58, [Traceback at line 2116](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2116)<br />2116..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 59, [Traceback at line 2143](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2143)<br />2143..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 60, [Traceback at line 2176](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2176)<br />2176..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 61, [Traceback at line 2203](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2203)<br />2203..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 62, [Traceback at line 2236](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2236)<br />2236..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 63, [Traceback at line 2263](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2263)<br />2263..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 64, [Traceback at line 2296](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2296)<br />2296..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 65, [Traceback at line 2323](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2323)<br />2323..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 66, [Traceback at line 2356](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2356)<br />2356..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 67, [Traceback at line 2383](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2383)<br />2383..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 68, [Traceback at line 2416](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2416)<br />2416..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 69, [Traceback at line 2443](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2443)<br />2443..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 70, [Traceback at line 2476](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2476)<br />2476..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 71, [Traceback at line 2503](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2503)<br />2503..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 72, [Traceback at line 2536](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2536)<br />2536..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 73, [Traceback at line 2563](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2563)<br />2563..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 74, [Traceback at line 2596](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2596)<br />2596..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 75, [Traceback at line 2623](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2623)<br />2623..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 76, [Traceback at line 2656](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2656)<br />2656..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 77, [Traceback at line 2683](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2683)<br />2683..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 78, [Traceback at line 2716](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2716)<br />2716..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 79, [Traceback at line 2743](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2743)<br />2743..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 80, [Traceback at line 2776](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2776)<br />2776..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 81, [Traceback at line 2803](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2803)<br />2803..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 82, [Traceback at line 2836](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2836)<br />2836..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 83, [Traceback at line 2863](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2863)<br />2863..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2896](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2896)<br />2896..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 85, [Traceback at line 2923](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2923)<br />2923..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 86, [Traceback at line 2956](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2956)<br />2956..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 87, [Traceback at line 2983](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2983)<br />2983..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 3016](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3016)<br />3016..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 89, [Traceback at line 3043](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3043)<br />3043..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 3076](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3076)<br />3076..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 91, [Traceback at line 3103](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3103)<br />3103..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 3136](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3136)<br />3136..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 93, [Traceback at line 3163](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3163)<br />3163..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 3196](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3196)<br />3196..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 95, [Traceback at line 3223](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3223)<br />3223..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 3256](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3256)<br />3256..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 97, [Traceback at line 3283](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3283)<br />3283..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 3316](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3316)<br />3316..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 99, [Traceback at line 3343](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3343)<br />3343..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 3376](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3376)<br />3376..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 101, [Traceback at line 3403](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3403)<br />3403..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 3436](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3436)<br />3436..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 103, [Traceback at line 3463](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3463)<br />3463..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 3496](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3496)<br />3496..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 105, [Traceback at line 3523](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3523)<br />3523..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

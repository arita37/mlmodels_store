## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 95](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L95)<br />95..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 106](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L106)<br />106..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 137](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L137)<br />137..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 263](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L263)<br />263..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 293](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L293)<br />293..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 310](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L310)<br />310..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 317](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L317)<br />317..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 338](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L338)<br />338..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 366](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L366)<br />366..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 385](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L385)<br />385..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 466](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L466)<br />466..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 497](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L497)<br />497..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 524](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L524)<br />524..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 552](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L552)<br />552..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 571](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L571)<br />571..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 599](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L599)<br />599..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 623](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L623)<br />623..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 642](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L642)<br />642..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 670](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L670)<br />670..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 689](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L689)<br />689..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 717](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L717)<br />717..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 736](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L736)<br />736..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 764](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L764)<br />764..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 788](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L788)<br />788..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 812](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L812)<br />812..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 831](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L831)<br />831..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 859](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L859)<br />859..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 883](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L883)<br />883..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 907](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L907)<br />907..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 931](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L931)<br />931..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 950](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L950)<br />950..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 969](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L969)<br />969..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 1005](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1005)<br />1005..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 34, [Traceback at line 1022](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1022)<br />1022..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 35, [Traceback at line 1029](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1029)<br />1029..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 36, [Traceback at line 1067](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1067)<br />1067..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 37, [Traceback at line 1084](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1084)<br />1084..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 38, [Traceback at line 1091](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1091)<br />1091..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 39, [Traceback at line 1129](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1129)<br />1129..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 40, [Traceback at line 1146](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1146)<br />1146..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 41, [Traceback at line 1153](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1153)<br />1153..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 42, [Traceback at line 1191](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1191)<br />1191..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 43, [Traceback at line 1208](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1208)<br />1208..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 44, [Traceback at line 1215](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1215)<br />1215..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 45, [Traceback at line 1253](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1253)<br />1253..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 46, [Traceback at line 1270](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1270)<br />1270..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 47, [Traceback at line 1277](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1277)<br />1277..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 48, [Traceback at line 1315](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1315)<br />1315..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 49, [Traceback at line 1332](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1332)<br />1332..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 50, [Traceback at line 1339](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1339)<br />1339..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 51, [Traceback at line 1377](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1377)<br />1377..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 52, [Traceback at line 1394](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1394)<br />1394..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 53, [Traceback at line 1401](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1401)<br />1401..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 54, [Traceback at line 1439](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1439)<br />1439..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 55, [Traceback at line 1456](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1456)<br />1456..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 56, [Traceback at line 1463](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1463)<br />1463..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 57, [Traceback at line 1506](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1506)<br />1506..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 58, [Traceback at line 1523](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1523)<br />1523..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 59, [Traceback at line 1530](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1530)<br />1530..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 60, [Traceback at line 1568](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1568)<br />1568..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 61, [Traceback at line 1585](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1585)<br />1585..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 62, [Traceback at line 1592](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1592)<br />1592..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 63, [Traceback at line 1630](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1630)<br />1630..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 64, [Traceback at line 1647](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1647)<br />1647..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 65, [Traceback at line 1654](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1654)<br />1654..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 66, [Traceback at line 1692](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1692)<br />1692..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 67, [Traceback at line 1709](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1709)<br />1709..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 68, [Traceback at line 1716](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1716)<br />1716..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 69, [Traceback at line 1754](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1754)<br />1754..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 70, [Traceback at line 1771](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1771)<br />1771..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 71, [Traceback at line 1778](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1778)<br />1778..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 72, [Traceback at line 1816](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1816)<br />1816..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 73, [Traceback at line 1833](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1833)<br />1833..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 74, [Traceback at line 1840](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1840)<br />1840..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 75, [Traceback at line 1878](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1878)<br />1878..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 76, [Traceback at line 1895](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1895)<br />1895..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 77, [Traceback at line 1902](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1902)<br />1902..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 78, [Traceback at line 1940](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1940)<br />1940..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_gluon/gluonts_model.py", line 54, in <module>
<br />    from mlmodels.util import load_function_uri
<br />ImportError: cannot import name 'load_function_uri'



### Error 79, [Traceback at line 1957](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1957)<br />1957..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 80, [Traceback at line 1964](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1964)<br />1964..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, cannot import name 'load_function_uri', tuple index out of range



### Error 81, [Traceback at line 1990](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1990)<br />1990..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 82, [Traceback at line 2014](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2014)<br />2014..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 83, [Traceback at line 2033](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2033)<br />2033..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2061](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2061)<br />2061..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 85, [Traceback at line 2085](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2085)<br />2085..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 86, [Traceback at line 2114](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2114)<br />2114..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 87, [Traceback at line 2141](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2141)<br />2141..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 2174](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2174)<br />2174..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 89, [Traceback at line 2201](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2201)<br />2201..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 2234](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2234)<br />2234..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 91, [Traceback at line 2261](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2261)<br />2261..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 2294](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2294)<br />2294..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 93, [Traceback at line 2321](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2321)<br />2321..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 2354](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2354)<br />2354..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 95, [Traceback at line 2381](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2381)<br />2381..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 2414](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2414)<br />2414..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 97, [Traceback at line 2441](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2441)<br />2441..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 2474](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2474)<br />2474..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 99, [Traceback at line 2501](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2501)<br />2501..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 2534](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2534)<br />2534..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 101, [Traceback at line 2561](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2561)<br />2561..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 2594](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2594)<br />2594..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 103, [Traceback at line 2621](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2621)<br />2621..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 2654](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2654)<br />2654..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 105, [Traceback at line 2681](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2681)<br />2681..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 106, [Traceback at line 2714](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2714)<br />2714..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 107, [Traceback at line 2741](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2741)<br />2741..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 108, [Traceback at line 2774](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2774)<br />2774..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 109, [Traceback at line 2801](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2801)<br />2801..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 110, [Traceback at line 2834](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2834)<br />2834..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 111, [Traceback at line 2861](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2861)<br />2861..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 112, [Traceback at line 2894](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2894)<br />2894..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 113, [Traceback at line 2921](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2921)<br />2921..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 114, [Traceback at line 2954](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2954)<br />2954..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 115, [Traceback at line 2981](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2981)<br />2981..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 116, [Traceback at line 3014](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3014)<br />3014..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 117, [Traceback at line 3041](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3041)<br />3041..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 118, [Traceback at line 3074](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3074)<br />3074..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 119, [Traceback at line 3101](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3101)<br />3101..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 120, [Traceback at line 3134](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3134)<br />3134..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 121, [Traceback at line 3161](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3161)<br />3161..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 122, [Traceback at line 3194](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3194)<br />3194..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 123, [Traceback at line 3221](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3221)<br />3221..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 124, [Traceback at line 3254](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3254)<br />3254..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 125, [Traceback at line 3281](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3281)<br />3281..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 126, [Traceback at line 3314](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3314)<br />3314..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 127, [Traceback at line 3341](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3341)<br />3341..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 128, [Traceback at line 3374](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3374)<br />3374..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 129, [Traceback at line 3401](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3401)<br />3401..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 130, [Traceback at line 3434](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3434)<br />3434..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 131, [Traceback at line 3461](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3461)<br />3461..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 132, [Traceback at line 3494](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3494)<br />3494..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 133, [Traceback at line 3521](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3521)<br />3521..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 134, [Traceback at line 3554](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3554)<br />3554..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 135, [Traceback at line 3581](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3581)<br />3581..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 136, [Traceback at line 3614](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3614)<br />3614..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 137, [Traceback at line 3641](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3641)<br />3641..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

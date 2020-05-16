## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py


### Error 1, [Traceback at line 91](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L91)<br />91..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 102](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L102)<br />102..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 133](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L133)<br />133..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 269](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L269)<br />269..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 299](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L299)<br />299..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 316](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L316)<br />316..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 323](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L323)<br />323..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 344](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L344)<br />344..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 372](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L372)<br />372..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 391](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L391)<br />391..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 472](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L472)<br />472..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 503](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L503)<br />503..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 418, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 530](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L530)<br />530..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 558](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L558)<br />558..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 577](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L577)<br />577..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 605](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L605)<br />605..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 629](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L629)<br />629..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 648](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L648)<br />648..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 676](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L676)<br />676..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 695](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L695)<br />695..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 723](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L723)<br />723..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 742](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L742)<br />742..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 770](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L770)<br />770..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 794](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L794)<br />794..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 818](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L818)<br />818..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 837](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L837)<br />837..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 865](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L865)<br />865..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 889](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L889)<br />889..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 913](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L913)<br />913..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 937](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L937)<br />937..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 956](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L956)<br />956..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 975](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L975)<br />975..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 996](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L996)<br />996..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 34, [Traceback at line 1026](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1026)<br />1026..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 35, [Traceback at line 1033](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1033)<br />1033..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 36, [Traceback at line 1056](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1056)<br />1056..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 37, [Traceback at line 1086](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1086)<br />1086..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 38, [Traceback at line 1093](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1093)<br />1093..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 39, [Traceback at line 1116](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1116)<br />1116..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 40, [Traceback at line 1146](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1146)<br />1146..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 41, [Traceback at line 1153](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1153)<br />1153..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 42, [Traceback at line 1176](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1176)<br />1176..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 43, [Traceback at line 1206](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1206)<br />1206..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 44, [Traceback at line 1213](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1213)<br />1213..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 45, [Traceback at line 1236](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1236)<br />1236..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 46, [Traceback at line 1266](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1266)<br />1266..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 47, [Traceback at line 1273](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1273)<br />1273..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 48, [Traceback at line 1296](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1296)<br />1296..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 49, [Traceback at line 1326](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1326)<br />1326..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 50, [Traceback at line 1333](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1333)<br />1333..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 51, [Traceback at line 1356](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1356)<br />1356..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 52, [Traceback at line 1386](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1386)<br />1386..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 53, [Traceback at line 1393](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1393)<br />1393..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 54, [Traceback at line 1416](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1416)<br />1416..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 55, [Traceback at line 1446](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1446)<br />1446..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 56, [Traceback at line 1453](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1453)<br />1453..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 57, [Traceback at line 1481](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1481)<br />1481..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 58, [Traceback at line 1511](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1511)<br />1511..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 59, [Traceback at line 1518](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1518)<br />1518..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 60, [Traceback at line 1541](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1541)<br />1541..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 61, [Traceback at line 1571](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1571)<br />1571..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 62, [Traceback at line 1578](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1578)<br />1578..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 63, [Traceback at line 1601](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1601)<br />1601..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 64, [Traceback at line 1631](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1631)<br />1631..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 65, [Traceback at line 1638](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1638)<br />1638..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 66, [Traceback at line 1661](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1661)<br />1661..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 67, [Traceback at line 1691](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1691)<br />1691..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 68, [Traceback at line 1698](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1698)<br />1698..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 69, [Traceback at line 1721](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1721)<br />1721..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 70, [Traceback at line 1751](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1751)<br />1751..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 71, [Traceback at line 1758](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1758)<br />1758..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 72, [Traceback at line 1781](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1781)<br />1781..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 73, [Traceback at line 1811](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1811)<br />1811..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 74, [Traceback at line 1818](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1818)<br />1818..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 75, [Traceback at line 1841](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1841)<br />1841..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 76, [Traceback at line 1871](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1871)<br />1871..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 77, [Traceback at line 1878](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1878)<br />1878..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 78, [Traceback at line 1901](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1901)<br />1901..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 79, [Traceback at line 1931](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1931)<br />1931..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 80, [Traceback at line 1938](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1938)<br />1938..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 81, [Traceback at line 1964](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1964)<br />1964..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 82, [Traceback at line 1988](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1988)<br />1988..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 83, [Traceback at line 2007](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2007)<br />2007..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2035](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2035)<br />2035..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 85, [Traceback at line 2059](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2059)<br />2059..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 86, [Traceback at line 2128](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2128)<br />2128..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 87, [Traceback at line 2163](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2163)<br />2163..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 2201](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2201)<br />2201..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 89, [Traceback at line 2236](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2236)<br />2236..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 2274](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2274)<br />2274..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 91, [Traceback at line 2309](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2309)<br />2309..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 2347](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2347)<br />2347..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 93, [Traceback at line 2382](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2382)<br />2382..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 2420](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2420)<br />2420..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 95, [Traceback at line 2455](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2455)<br />2455..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 2493](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2493)<br />2493..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 97, [Traceback at line 2528](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2528)<br />2528..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 2566](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2566)<br />2566..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 99, [Traceback at line 2601](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2601)<br />2601..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 2639](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2639)<br />2639..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 101, [Traceback at line 2674](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2674)<br />2674..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 2712](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2712)<br />2712..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 103, [Traceback at line 2747](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2747)<br />2747..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 2785](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2785)<br />2785..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 105, [Traceback at line 2820](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2820)<br />2820..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 106, [Traceback at line 2858](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2858)<br />2858..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 107, [Traceback at line 2893](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2893)<br />2893..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 108, [Traceback at line 2931](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2931)<br />2931..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 109, [Traceback at line 2966](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L2966)<br />2966..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 110, [Traceback at line 3004](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3004)<br />3004..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 111, [Traceback at line 3039](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3039)<br />3039..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 112, [Traceback at line 3077](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3077)<br />3077..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 113, [Traceback at line 3112](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3112)<br />3112..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 114, [Traceback at line 3150](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3150)<br />3150..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 115, [Traceback at line 3185](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3185)<br />3185..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 116, [Traceback at line 3223](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3223)<br />3223..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 117, [Traceback at line 3258](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3258)<br />3258..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 118, [Traceback at line 3296](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3296)<br />3296..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 119, [Traceback at line 3331](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3331)<br />3331..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 120, [Traceback at line 3369](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3369)<br />3369..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 121, [Traceback at line 3404](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3404)<br />3404..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 122, [Traceback at line 3442](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3442)<br />3442..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 123, [Traceback at line 3477](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3477)<br />3477..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 124, [Traceback at line 3515](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3515)<br />3515..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 125, [Traceback at line 3550](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3550)<br />3550..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 126, [Traceback at line 3588](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3588)<br />3588..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 127, [Traceback at line 3623](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3623)<br />3623..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 128, [Traceback at line 3661](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3661)<br />3661..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 129, [Traceback at line 3696](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3696)<br />3696..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 130, [Traceback at line 3734](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3734)<br />3734..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 131, [Traceback at line 3769](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3769)<br />3769..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 132, [Traceback at line 3807](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3807)<br />3807..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 133, [Traceback at line 3842](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3842)<br />3842..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 134, [Traceback at line 3880](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3880)<br />3880..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 135, [Traceback at line 3915](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3915)<br />3915..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 136, [Traceback at line 3953](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3953)<br />3953..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 137, [Traceback at line 3988](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-17-07_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L3988)<br />3988..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py


### Error 1, [Traceback at line 91](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L91)<br />91..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 102](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L102)<br />102..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 133](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L133)<br />133..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 258](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L258)<br />258..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 288](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L288)<br />288..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 305](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L305)<br />305..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 312](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L312)<br />312..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 333](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L333)<br />333..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 361](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L361)<br />361..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 380](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L380)<br />380..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 461](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L461)<br />461..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 492](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L492)<br />492..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 418, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 519](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L519)<br />519..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 547](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L547)<br />547..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 566](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L566)<br />566..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 594](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L594)<br />594..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 618](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L618)<br />618..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 637](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L637)<br />637..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 665](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L665)<br />665..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 684](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L684)<br />684..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 712](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L712)<br />712..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 731](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L731)<br />731..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 759](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L759)<br />759..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 783](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L783)<br />783..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 807](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L807)<br />807..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 826](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L826)<br />826..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 854](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L854)<br />854..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 878](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L878)<br />878..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 902](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L902)<br />902..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 926](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L926)<br />926..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 945](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L945)<br />945..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 964](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L964)<br />964..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 985](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L985)<br />985..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 34, [Traceback at line 1015](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1015)<br />1015..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 35, [Traceback at line 1022](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1022)<br />1022..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 36, [Traceback at line 1045](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1045)<br />1045..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 37, [Traceback at line 1075](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1075)<br />1075..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 38, [Traceback at line 1082](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1082)<br />1082..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 39, [Traceback at line 1105](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1105)<br />1105..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 40, [Traceback at line 1135](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1135)<br />1135..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 41, [Traceback at line 1142](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1142)<br />1142..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 42, [Traceback at line 1165](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1165)<br />1165..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 43, [Traceback at line 1195](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1195)<br />1195..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 44, [Traceback at line 1202](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1202)<br />1202..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 45, [Traceback at line 1225](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1225)<br />1225..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 46, [Traceback at line 1255](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1255)<br />1255..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 47, [Traceback at line 1262](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1262)<br />1262..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 48, [Traceback at line 1285](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1285)<br />1285..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 49, [Traceback at line 1315](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1315)<br />1315..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 50, [Traceback at line 1322](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1322)<br />1322..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 51, [Traceback at line 1345](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1345)<br />1345..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 52, [Traceback at line 1375](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1375)<br />1375..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 53, [Traceback at line 1382](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1382)<br />1382..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 54, [Traceback at line 1405](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1405)<br />1405..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 55, [Traceback at line 1435](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1435)<br />1435..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 56, [Traceback at line 1442](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1442)<br />1442..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 57, [Traceback at line 1470](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1470)<br />1470..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 58, [Traceback at line 1500](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1500)<br />1500..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 59, [Traceback at line 1507](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1507)<br />1507..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 60, [Traceback at line 1530](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1530)<br />1530..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 61, [Traceback at line 1560](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1560)<br />1560..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 62, [Traceback at line 1567](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1567)<br />1567..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 63, [Traceback at line 1590](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1590)<br />1590..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 64, [Traceback at line 1620](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1620)<br />1620..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 65, [Traceback at line 1627](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1627)<br />1627..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 66, [Traceback at line 1650](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1650)<br />1650..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 67, [Traceback at line 1680](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1680)<br />1680..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 68, [Traceback at line 1687](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1687)<br />1687..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 69, [Traceback at line 1710](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1710)<br />1710..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 70, [Traceback at line 1740](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1740)<br />1740..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 71, [Traceback at line 1747](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1747)<br />1747..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 72, [Traceback at line 1770](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1770)<br />1770..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 73, [Traceback at line 1800](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1800)<br />1800..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 74, [Traceback at line 1807](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1807)<br />1807..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 75, [Traceback at line 1830](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1830)<br />1830..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 76, [Traceback at line 1860](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1860)<br />1860..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 77, [Traceback at line 1867](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1867)<br />1867..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 78, [Traceback at line 1890](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1890)<br />1890..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 79, [Traceback at line 1920](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1920)<br />1920..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 80, [Traceback at line 1927](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1927)<br />1927..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 81, [Traceback at line 1953](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1953)<br />1953..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 82, [Traceback at line 1977](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1977)<br />1977..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 83, [Traceback at line 1996](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L1996)<br />1996..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2024](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2024)<br />2024..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 85, [Traceback at line 2048](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2048)<br />2048..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 86, [Traceback at line 2107](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2107)<br />2107..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 87, [Traceback at line 2142](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2142)<br />2142..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 2180](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2180)<br />2180..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 89, [Traceback at line 2215](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2215)<br />2215..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 2253](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2253)<br />2253..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 91, [Traceback at line 2288](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2288)<br />2288..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 2326](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2326)<br />2326..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 93, [Traceback at line 2361](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2361)<br />2361..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 2399](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2399)<br />2399..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 95, [Traceback at line 2434](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2434)<br />2434..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 2472](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2472)<br />2472..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 97, [Traceback at line 2507](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2507)<br />2507..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 2545](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2545)<br />2545..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 99, [Traceback at line 2580](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2580)<br />2580..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 2618](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2618)<br />2618..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 101, [Traceback at line 2653](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2653)<br />2653..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 2691](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2691)<br />2691..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 103, [Traceback at line 2726](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2726)<br />2726..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 2764](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2764)<br />2764..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 105, [Traceback at line 2799](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2799)<br />2799..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 106, [Traceback at line 2837](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2837)<br />2837..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 107, [Traceback at line 2872](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2872)<br />2872..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 108, [Traceback at line 2910](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2910)<br />2910..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 109, [Traceback at line 2945](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2945)<br />2945..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 110, [Traceback at line 2983](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L2983)<br />2983..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 111, [Traceback at line 3018](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3018)<br />3018..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 112, [Traceback at line 3056](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3056)<br />3056..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 113, [Traceback at line 3091](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3091)<br />3091..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 114, [Traceback at line 3129](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3129)<br />3129..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 115, [Traceback at line 3164](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3164)<br />3164..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 116, [Traceback at line 3202](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3202)<br />3202..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 117, [Traceback at line 3237](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3237)<br />3237..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 118, [Traceback at line 3275](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3275)<br />3275..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 119, [Traceback at line 3310](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3310)<br />3310..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 120, [Traceback at line 3348](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3348)<br />3348..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 121, [Traceback at line 3383](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3383)<br />3383..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 122, [Traceback at line 3421](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3421)<br />3421..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 123, [Traceback at line 3456](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3456)<br />3456..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 124, [Traceback at line 3494](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3494)<br />3494..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 125, [Traceback at line 3529](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3529)<br />3529..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 126, [Traceback at line 3567](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3567)<br />3567..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 127, [Traceback at line 3602](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3602)<br />3602..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 128, [Traceback at line 3640](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3640)<br />3640..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 129, [Traceback at line 3675](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3675)<br />3675..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 130, [Traceback at line 3713](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3713)<br />3713..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 131, [Traceback at line 3748](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3748)<br />3748..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 132, [Traceback at line 3786](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3786)<br />3786..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 133, [Traceback at line 3821](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3821)<br />3821..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 134, [Traceback at line 3859](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3859)<br />3859..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 135, [Traceback at line 3894](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3894)<br />3894..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 136, [Traceback at line 3932](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3932)<br />3932..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/model_tch/torchhub.py", line 46, in _train
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



### Error 137, [Traceback at line 3967](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json_2020-05-15-15-29_4c47a0bacf53fc18eb078111d70311c747fed1fc.py#L3967)<br />3967..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 95](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L95)<br />95..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 106](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L106)<br />106..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 137](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L137)<br />137..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 262](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L262)<br />262..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 292](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L292)<br />292..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 309](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L309)<br />309..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 316](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L316)<br />316..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 337](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L337)<br />337..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 365](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L365)<br />365..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 384](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L384)<br />384..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 465](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L465)<br />465..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 496](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L496)<br />496..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 523](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L523)<br />523..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 551](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L551)<br />551..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 570](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L570)<br />570..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 598](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L598)<br />598..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 622](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L622)<br />622..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 641](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L641)<br />641..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 669](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L669)<br />669..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 688](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L688)<br />688..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 716](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L716)<br />716..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 735](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L735)<br />735..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 763](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L763)<br />763..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 787](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L787)<br />787..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 811](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L811)<br />811..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 830](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L830)<br />830..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 858](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L858)<br />858..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 882](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L882)<br />882..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 906](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L906)<br />906..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 930](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L930)<br />930..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 949](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L949)<br />949..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 968](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L968)<br />968..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 1022](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1022)<br />1022..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 34, [Traceback at line 1076](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1076)<br />1076..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 35, [Traceback at line 1135](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1135)<br />1135..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 36, [Traceback at line 1189](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1189)<br />1189..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 37, [Traceback at line 1253](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1253)<br />1253..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 38, [Traceback at line 1307](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1307)<br />1307..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 39, [Traceback at line 1360](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1360)<br />1360..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 40, [Traceback at line 1396](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1396)<br />1396..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json  --config_mode deepar  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />  ##### Init model_gluon.gluonts_model {'path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_gluon/gluonts_deepar/', 'model_uri': 'model_gluon.gluonts_model'} 
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
<br />  ##### Fit <mlmodels.model_gluon.gluonts_model.Model object at 0x7fc412870d68> 
<br />INFO:root:Start model training
<br />INFO:root:Epoch[0] Learning rate is 0.001
<br />
<br />  0%|          | 0/10 [00:00<?, ?it/s]INFO:numexpr.utils:NumExpr defaulting to 2 threads.
<br />INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
<br />
<br />100%|██████████| 10/10 [00:02<00:00,  3.75it/s, avg_epoch_loss=5.22]
<br />INFO:root:Epoch[0] Elapsed time 2.669 seconds
<br />INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.220498
<br />INFO:root:Loading parameters from best epoch (0)
<br />INFO:root:Final loss: 5.220497798919678 (occurred at epoch 0)
<br />INFO:root:End model training
<br />[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
<br />{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
<br />learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.



### Error 41, [Traceback at line 1464](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1464)<br />1464..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 42, [Traceback at line 1518](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1518)<br />1518..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 43, [Traceback at line 1577](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1577)<br />1577..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 44, [Traceback at line 1631](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1631)<br />1631..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 45, [Traceback at line 1695](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1695)<br />1695..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 46, [Traceback at line 1749](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1749)<br />1749..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 47, [Traceback at line 1802](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1802)<br />1802..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 48, [Traceback at line 1838](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1838)<br />1838..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 421, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json  --config_mode test  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 



### Error 49, [Traceback at line 1871](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1871)<br />1871..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 50, [Traceback at line 1895](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1895)<br />1895..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 51, [Traceback at line 1914](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1914)<br />1914..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 297, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 52, [Traceback at line 1942](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1942)<br />1942..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 415, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 53, [Traceback at line 1966](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1966)<br />1966..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 414, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 54, [Traceback at line 1995](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1995)<br />1995..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 55, [Traceback at line 2022](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2022)<br />2022..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 56, [Traceback at line 2055](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2055)<br />2055..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 57, [Traceback at line 2082](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2082)<br />2082..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 58, [Traceback at line 2115](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2115)<br />2115..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 59, [Traceback at line 2142](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2142)<br />2142..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 60, [Traceback at line 2175](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2175)<br />2175..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 61, [Traceback at line 2202](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2202)<br />2202..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 62, [Traceback at line 2235](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2235)<br />2235..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 63, [Traceback at line 2262](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2262)<br />2262..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 64, [Traceback at line 2295](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2295)<br />2295..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 65, [Traceback at line 2322](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2322)<br />2322..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 66, [Traceback at line 2355](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2355)<br />2355..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 67, [Traceback at line 2382](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2382)<br />2382..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 68, [Traceback at line 2415](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2415)<br />2415..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 69, [Traceback at line 2442](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2442)<br />2442..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 70, [Traceback at line 2475](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2475)<br />2475..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 71, [Traceback at line 2502](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2502)<br />2502..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 72, [Traceback at line 2535](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2535)<br />2535..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 73, [Traceback at line 2562](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2562)<br />2562..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 74, [Traceback at line 2595](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2595)<br />2595..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 75, [Traceback at line 2622](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2622)<br />2622..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 76, [Traceback at line 2655](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2655)<br />2655..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 77, [Traceback at line 2682](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2682)<br />2682..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 78, [Traceback at line 2715](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2715)<br />2715..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 79, [Traceback at line 2742](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2742)<br />2742..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 80, [Traceback at line 2775](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2775)<br />2775..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 81, [Traceback at line 2802](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2802)<br />2802..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 82, [Traceback at line 2835](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2835)<br />2835..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 83, [Traceback at line 2862](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2862)<br />2862..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2895](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2895)<br />2895..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 85, [Traceback at line 2922](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2922)<br />2922..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 86, [Traceback at line 2955](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2955)<br />2955..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 87, [Traceback at line 2982](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2982)<br />2982..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 3015](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3015)<br />3015..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 89, [Traceback at line 3042](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3042)<br />3042..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 3075](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3075)<br />3075..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 91, [Traceback at line 3102](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3102)<br />3102..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 3135](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3135)<br />3135..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 93, [Traceback at line 3162](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3162)<br />3162..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 3195](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3195)<br />3195..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 95, [Traceback at line 3222](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3222)<br />3222..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 3255](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3255)<br />3255..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 97, [Traceback at line 3282](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3282)<br />3282..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 3315](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3315)<br />3315..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 99, [Traceback at line 3342](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3342)<br />3342..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 3375](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3375)<br />3375..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 101, [Traceback at line 3402](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3402)<br />3402..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 3435](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3435)<br />3435..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 103, [Traceback at line 3462](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3462)<br />3462..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 3495](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3495)<br />3495..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataloader.py", line 236, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 105, [Traceback at line 3522](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3522)<br />3522..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 413, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 299, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

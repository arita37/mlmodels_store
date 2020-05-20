## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py


### Error 1, [Traceback at line 93](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L93)<br />93..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 2, [Traceback at line 104](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L104)<br />104..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 3, [Traceback at line 135](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L135)<br />135..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 4, [Traceback at line 271](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L271)<br />271..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 5, [Traceback at line 301](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L301)<br />301..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 6, [Traceback at line 318](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L318)<br />318..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 325](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L325)<br />325..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 417, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 8, [Traceback at line 346](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L346)<br />346..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 9, [Traceback at line 374](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L374)<br />374..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 10, [Traceback at line 393](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L393)<br />393..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 11, [Traceback at line 474](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L474)<br />474..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 12, [Traceback at line 505](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L505)<br />505..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 418, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/text/quora/train.csv'



### Error 13, [Traceback at line 532](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L532)<br />532..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 14, [Traceback at line 560](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L560)<br />560..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 15, [Traceback at line 579](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L579)<br />579..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 16, [Traceback at line 607](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L607)<br />607..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 17, [Traceback at line 631](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L631)<br />631..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 18, [Traceback at line 650](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L650)<br />650..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 19, [Traceback at line 678](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L678)<br />678..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 20, [Traceback at line 697](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L697)<br />697..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 21, [Traceback at line 725](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L725)<br />725..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 22, [Traceback at line 744](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L744)<br />744..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 23, [Traceback at line 772](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L772)<br />772..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 24, [Traceback at line 796](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L796)<br />796..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 25, [Traceback at line 820](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L820)<br />820..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 26, [Traceback at line 839](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L839)<br />839..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 27, [Traceback at line 867](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L867)<br />867..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 28, [Traceback at line 891](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L891)<br />891..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 29, [Traceback at line 915](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L915)<br />915..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 30, [Traceback at line 939](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L939)<br />939..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 31, [Traceback at line 958](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L958)<br />958..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 32, [Traceback at line 977](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L977)<br />977..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 33, [Traceback at line 1031](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1031)<br />1031..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 34, [Traceback at line 1085](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1085)<br />1085..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 35, [Traceback at line 1144](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1144)<br />1144..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 36, [Traceback at line 1198](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1198)<br />1198..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 37, [Traceback at line 1262](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1262)<br />1262..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 38, [Traceback at line 1316](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1316)<br />1316..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 39, [Traceback at line 1369](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1369)<br />1369..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 40, [Traceback at line 1405](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1405)<br />1405..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 418, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json  --config_mode deepar  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/json/benchmark_timeseries/test02/model_list_gluon_only.json 
<br />
<br />  ##### Init model_gluon.gluonts_model {'path': 'https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/ztest/model_gluon/gluonts_deepar/', 'model_uri': 'model_gluon.gluonts_model'} 
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
<br />  ##### Fit <mlmodels.model_gluon.gluonts_model.Model object at 0x7f68af2efd68> 
<br />INFO:root:Start model training
<br />INFO:root:Epoch[0] Learning rate is 0.001
<br />
<br />  0%|          | 0/10 [00:00<?, ?it/s]INFO:numexpr.utils:NumExpr defaulting to 2 threads.
<br />INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
<br />
<br />100%|██████████| 10/10 [00:02<00:00,  3.96it/s, avg_epoch_loss=5.24]
<br />INFO:root:Epoch[0] Elapsed time 2.530 seconds
<br />INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.244711
<br />INFO:root:Loading parameters from best epoch (0)
<br />INFO:root:Final loss: 5.2447114944458 (occurred at epoch 0)
<br />INFO:root:End model training
<br />[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
<br />{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
<br />learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.



### Error 41, [Traceback at line 1473](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1473)<br />1473..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 42, [Traceback at line 1527](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1527)<br />1527..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 43, [Traceback at line 1586](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1586)<br />1586..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 44, [Traceback at line 1640](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1640)<br />1640..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 45, [Traceback at line 1704](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1704)<br />1704..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 46, [Traceback at line 1758](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1758)<br />1758..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 47, [Traceback at line 1811](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1811)<br />1811..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: 'Model' object is not iterable



### Error 48, [Traceback at line 1847](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1847)<br />1847..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 418, in fit_cli
<br />    model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 113, in model_create
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br /> ************ JSON File https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  ml_models --do fit --config_file https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json  --config_mode test  
<br />fit
<br />
<br />  ##### Load JSON https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataset/json/benchmark_timeseries/test01/armdn.json 



### Error 49, [Traceback at line 1880](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1880)<br />1880..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 50, [Traceback at line 1904](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1904)<br />1904..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 51, [Traceback at line 1923](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1923)<br />1923..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 294, in config_get_pars
<br />    data_p    = path_norm_dict( js.get("data_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 52, [Traceback at line 1951](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1951)<br />1951..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 412, in fit_cli
<br />    path      = out_p['path']
<br />KeyError: 'path'



### Error 53, [Traceback at line 1975](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L1975)<br />1975..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 411, in fit_cli
<br />    model_uri = model_p['model_uri']
<br />KeyError: 'model_uri'



### Error 54, [Traceback at line 2004](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2004)<br />2004..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 55, [Traceback at line 2031](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2031)<br />2031..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 56, [Traceback at line 2064](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2064)<br />2064..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 57, [Traceback at line 2091](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2091)<br />2091..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 58, [Traceback at line 2124](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2124)<br />2124..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 59, [Traceback at line 2151](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2151)<br />2151..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 60, [Traceback at line 2184](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2184)<br />2184..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 61, [Traceback at line 2211](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2211)<br />2211..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 62, [Traceback at line 2244](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2244)<br />2244..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 63, [Traceback at line 2271](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2271)<br />2271..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 64, [Traceback at line 2304](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2304)<br />2304..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 65, [Traceback at line 2331](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2331)<br />2331..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 66, [Traceback at line 2364](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2364)<br />2364..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 67, [Traceback at line 2391](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2391)<br />2391..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 68, [Traceback at line 2424](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2424)<br />2424..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 69, [Traceback at line 2451](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2451)<br />2451..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 70, [Traceback at line 2484](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2484)<br />2484..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 71, [Traceback at line 2511](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2511)<br />2511..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 72, [Traceback at line 2544](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2544)<br />2544..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 73, [Traceback at line 2571](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2571)<br />2571..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 74, [Traceback at line 2604](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2604)<br />2604..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 75, [Traceback at line 2631](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2631)<br />2631..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 76, [Traceback at line 2664](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2664)<br />2664..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 77, [Traceback at line 2691](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2691)<br />2691..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 78, [Traceback at line 2724](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2724)<br />2724..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 79, [Traceback at line 2751](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2751)<br />2751..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 80, [Traceback at line 2784](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2784)<br />2784..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 81, [Traceback at line 2811](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2811)<br />2811..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 82, [Traceback at line 2844](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2844)<br />2844..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 83, [Traceback at line 2871](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2871)<br />2871..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 84, [Traceback at line 2904](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2904)<br />2904..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 85, [Traceback at line 2931](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2931)<br />2931..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 86, [Traceback at line 2964](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2964)<br />2964..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 87, [Traceback at line 2991](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L2991)<br />2991..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 88, [Traceback at line 3024](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3024)<br />3024..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 89, [Traceback at line 3051](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3051)<br />3051..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 90, [Traceback at line 3084](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3084)<br />3084..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 91, [Traceback at line 3111](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3111)<br />3111..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 92, [Traceback at line 3144](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3144)<br />3144..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 93, [Traceback at line 3171](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3171)<br />3171..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 94, [Traceback at line 3204](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3204)<br />3204..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 95, [Traceback at line 3231](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3231)<br />3231..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 96, [Traceback at line 3264](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3264)<br />3264..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 97, [Traceback at line 3291](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3291)<br />3291..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 98, [Traceback at line 3324](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3324)<br />3324..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 99, [Traceback at line 3351](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3351)<br />3351..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 100, [Traceback at line 3384](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3384)<br />3384..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 101, [Traceback at line 3411](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3411)<br />3411..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 102, [Traceback at line 3444](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3444)<br />3444..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 103, [Traceback at line 3471](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3471)<br />3471..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'



### Error 104, [Traceback at line 3504](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3504)<br />3504..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 421, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 125, in fit
<br />    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/dataloader.py", line 238, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 105, [Traceback at line 3531](https://github.com/arita37/mlmodels_store/blob/master/log_json/log_json.py#L3531)<br />3531..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 527, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 410, in fit_cli
<br />    model_p, data_p, compute_p, out_p = config_get_pars(config_file, arg.config_mode)
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/models.py", line 296, in config_get_pars
<br />    out_p     = path_norm_dict( js.get("out_pars") )
<br />  File "https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679/mlmodels/util.py", line 201, in path_norm_dict
<br />    for k,v in ddict.items():
<br />AttributeError: 'NoneType' object has no attribute 'items'

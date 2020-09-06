## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 106](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L106)<br />106..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 424, in fit_cli
<br />    model, sess = fit(module, model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
<br />TypeError: fit() got multiple values for argument 'data_pars'



### Error 2, [Traceback at line 132](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L132)<br />132..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 534, in main
<br />    predict_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 445, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_tf/1_lstm.py", line 223, in load
<br />    d = pickle.load( open(path, mode="rb")  )
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest_1lstm//model/model_pars.pkl'



### Error 3, [Traceback at line 169](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L169)<br />169..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 526, in main
<br />    test_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 456, in test_cli
<br />    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 266, in test_module
<br />    model, sess = module.fit(model, data_pars = data_pars, compute_pars = compute_pars, out_pars = out_pars)
<br />TypeError: fit() got multiple values for argument 'data_pars'



### Error 4, [Traceback at line 191](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L191)<br />191..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 5, [Traceback at line 211](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L211)<br />211..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 218](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L218)<br />218..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 7, [Traceback at line 240](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L240)<br />240..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 8, [Traceback at line 260](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L260)<br />260..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 267](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L267)<br />267..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 530, in main
<br />    fit_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 420, in fit_cli
<br />    module = module_load(model_uri)  # '1_lstm.py
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 10, [Traceback at line 526](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L526)<br />526..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 11, [Traceback at line 532](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L532)<br />532..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 57, in optim
<br />    out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 170, in optim_optuna
<br />    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 302, in optimize
<br />    gc_after_trial, None)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 538, in _optimize_sequential
<br />    self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 550, in _run_trial_and_callbacks
<br />    trial = self._run_trial(func, catch, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 12, [Traceback at line 571](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L571)<br />571..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 13, [Traceback at line 577](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L577)<br />577..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 57, in optim
<br />    out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 170, in optim_optuna
<br />    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 302, in optimize
<br />    gc_after_trial, None)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 538, in _optimize_sequential
<br />    self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 550, in _run_trial_and_callbacks
<br />    trial = self._run_trial(func, catch, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 14, [Traceback at line 614](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L614)<br />614..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 15, [Traceback at line 620](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L620)<br />620..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 382, in main
<br />    test_json( path_json="template/optim_config_prune.json", config_mode= arg.config_mode )
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 230, in test_json
<br />    out_pars        = out_pars
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 57, in optim
<br />    out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 170, in optim_optuna
<br />    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 302, in optimize
<br />    gc_after_trial, None)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 538, in _optimize_sequential
<br />    self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 550, in _run_trial_and_callbacks
<br />    trial = self._run_trial(func, catch, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 16, [Traceback at line 859](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L859)<br />859..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_keras/armdn.py", line 35, in <module>
<br />    from mlmodels.data import (download_data, import_data)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/data.py", line 126
<br />    """   
<br />      if not target:
<br />         tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
<br />         os.makedirs(tmp)
<br />         target = tmp
<br />      """
<br />          
<br />                    
<br />                                                                         
<br />                         
<br />                     
<br />        ^
<br />SyntaxError: invalid syntax



### Error 17, [Traceback at line 889](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L889)<br />889..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 18, [Traceback at line 896](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L896)<br />896..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, invalid syntax (data.py, line 126), tuple index out of range



### Error 19, [Traceback at line 902](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L902)<br />902..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 20, [Traceback at line 914](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L914)<br />914..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 21, [Traceback at line 921](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L921)<br />921..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 22, [Traceback at line 927](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L927)<br />927..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 23, [Traceback at line 947](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L947)<br />947..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 24, [Traceback at line 954](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L954)<br />954..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 25, [Traceback at line 960](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L960)<br />960..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 26, [Traceback at line 980](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L980)<br />980..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 27, [Traceback at line 987](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L987)<br />987..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 28, [Traceback at line 993](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L993)<br />993..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />
<br />  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 29, [Traceback at line 1065](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1065)<br />1065..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 30, [Traceback at line 1072](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1072)<br />1072..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 31, [Traceback at line 1078](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1078)<br />1078..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 32, [Traceback at line 1098](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1098)<br />1098..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 33, [Traceback at line 1105](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1105)<br />1105..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 34, [Traceback at line 1111](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1111)<br />1111..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 35, [Traceback at line 1131](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1131)<br />1131..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 36, [Traceback at line 1138](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1138)<br />1138..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 37, [Traceback at line 1144](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1144)<br />1144..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 38, [Traceback at line 1164](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1164)<br />1164..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 39, [Traceback at line 1171](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1171)<br />1171..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 40, [Traceback at line 1177](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1177)<br />1177..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />
<br />  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/example/benchmark/ 
<br />
<br />                       date_run  ...            metric_name
<br />0  2020-09-06 00:56:02.401333  ...    mean_absolute_error
<br />1  2020-09-06 00:56:02.406047  ...     mean_squared_error
<br />2  2020-09-06 00:56:02.409665  ...  median_absolute_error
<br />3  2020-09-06 00:56:02.413301  ...               r2_score
<br />
<br />[4 rows x 6 columns] 
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 41, [Traceback at line 1222](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1222)<br />1222..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 42, [Traceback at line 1229](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1229)<br />1229..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 43, [Traceback at line 1235](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1235)<br />1235..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_gluon/gluonts_model.py", line 203
<br />    if d ==  "single_dataframe" :
<br />                                ^
<br />SyntaxError: invalid syntax



### Error 44, [Traceback at line 1255](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1255)<br />1255..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 45, [Traceback at line 1262](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1262)<br />1262..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, invalid syntax (gluonts_model.py, line 203), tuple index out of range



### Error 46, [Traceback at line 1324](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1324)<br />1324..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'



### Error 47, [Traceback at line 1329](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1329)<br />1329..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/model_keras/armdn.py", line 35, in <module>
<br />    from mlmodels.data import (download_data, import_data)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/data.py", line 126
<br />    """   
<br />      if not target:
<br />         tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
<br />         os.makedirs(tmp)
<br />         target = tmp
<br />      """
<br />          
<br />                    
<br />                                                                         
<br />                         
<br />                     
<br />        ^
<br />SyntaxError: invalid syntax



### Error 48, [Traceback at line 1359](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1359)<br />1359..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 49, [Traceback at line 1366](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1366)<br />1366..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, invalid syntax (data.py, line 126), tuple index out of range



### Error 50, [Traceback at line 1517](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1517)<br />1517..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 51, [Traceback at line 1529](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1529)<br />1529..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 52, [Traceback at line 1536](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L1536)<br />1536..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range

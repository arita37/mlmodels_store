## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py


### Error 1, [Traceback at line 191](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py#L191)<br />191..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 2, [Traceback at line 197](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py#L197)<br />197..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 382, in main
<br />    test_json( path_json="template/optim_config_prune.json", config_mode= arg.config_mode )
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 230, in test_json
<br />    out_pars        = out_pars
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 57, in optim
<br />    out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 170, in optim_optuna
<br />    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 302, in optimize
<br />    gc_after_trial, None)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 538, in _optimize_sequential
<br />    self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 550, in _run_trial_and_callbacks
<br />    trial = self._run_trial(func, catch, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 3, [Traceback at line 244](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py#L244)<br />244..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 4, [Traceback at line 250](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py#L250)<br />250..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 394, in <module>
<br />    main()
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 382, in main
<br />    test_json( path_json="template/optim_config_prune.json", config_mode= arg.config_mode )
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 230, in test_json
<br />    out_pars        = out_pars
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 57, in optim
<br />    out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 170, in optim_optuna
<br />    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 302, in optimize
<br />    gc_after_trial, None)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 538, in _optimize_sequential
<br />    self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 550, in _run_trial_and_callbacks
<br />    trial = self._run_trial(func, catch, gc_after_trial)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
<br />    result = func(trial)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/optim.py", line 143, in objective
<br />    module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
<br />UnboundLocalError: local variable 'module' referenced before assignment



### Error 5, [Traceback at line 283](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py#L283)<br />283..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/model_keras/textcnn.py", line 258, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/model_keras/textcnn.py", line 165, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 6, [Traceback at line 291](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-03-12-24_3fc653a7999aa2761b089e07ab0db303a5051c0b.py#L291)<br />291..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/ztest.py", line 655, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b/mlmodels/ztest.py", line 424, in test_pullrequest
<br />    raise Exception(f"Unknown dataset type", x)
<br />Exception: ('Unknown dataset type', '[W 2020-09-03 12:25:01,765] Setting status of trial#0 as TrialState.FAIL because of the following error: UnboundLocalError("local variable \'module\' referenced before assignment",)\n')

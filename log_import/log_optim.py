Deprecaton set to False

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[W 2020-08-29 06:01:05,840] Setting status of trial#0 as TrialState.FAIL because of the following error: TypeError("fit() got multiple values for argument 'data_pars'",)
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
    result = func(trial)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 146, in objective
    model, sess = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: fit() got multiple values for argument 'data_pars'
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 384, in main
    test_json( path_json="template/optim_config_prune.json", config_mode= arg.config_mode )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 232, in test_json
    out_pars        = out_pars
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 57, in optim
    out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 172, in optim_optuna
    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/optuna/study.py", line 302, in optimize
    gc_after_trial, None)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/optuna/study.py", line 538, in _optimize_sequential
    self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/optuna/study.py", line 550, in _run_trial_and_callbacks
    trial = self._run_trial(func, catch, gc_after_trial)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/optuna/study.py", line 569, in _run_trial
    result = func(trial)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 146, in objective
    model, sess = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: fit() got multiple values for argument 'data_pars'

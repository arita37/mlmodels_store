## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 108](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L108)<br />108..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c/mlmodels/model_sklearn//model_sklearn.py", line 259, in <module>
<br />    test_api(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c/mlmodels/models.py", line 230, in test_api
<br />    module, model = module_load_full(model_uri, model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c/mlmodels/models.py", line 101, in module_load_full
<br />    module.init(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />AttributeError: module 'mlmodels.model_sklearn.model_sklearn' has no attribute 'init'



### Error 2, [Traceback at line 116](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L116)<br />116..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c/mlmodels/ztest.py", line 655, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c/mlmodels/ztest.py", line 509, in test_all
<br />    log_remote_push()
<br />  File "https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c/mlmodels/ztest.py", line 154, in log_remote_push
<br />    tag = "m_" + str(arg.name)
<br />AttributeError: 'NoneType' object has no attribute 'name'

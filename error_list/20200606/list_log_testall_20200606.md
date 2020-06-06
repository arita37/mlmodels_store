## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 43](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L43)<br />43..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_keras//charcnn_zhang.py", line 261, in <module>
<br />    test(pars_choice="json", data_path=f"dataset/json/refactor/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_keras//charcnn_zhang.py", line 222, in test
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_keras//charcnn_zhang.py", line 151, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataset/json/refactor/charcnn_zhang.json'



### Error 2, [Traceback at line 51](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L51)<br />51..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/ztest.py", line 638, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/ztest.py", line 505, in test_all
<br />    log_remote_push()
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/ztest.py", line 154, in log_remote_push
<br />    tag = "m_" + str(arg.name)
<br />AttributeError: 'NoneType' object has no attribute 'name'

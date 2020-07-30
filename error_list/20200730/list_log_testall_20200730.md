## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 42](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L42)<br />42..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras//charcnn.py", line 373, in <module>
<br />    test(pars_choice="json", data_path= f"dataset/json/refactor/charcnn.json")
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras//charcnn.py", line 330, in test
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/model_keras//charcnn.py", line 266, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/dataset/json/refactor/charcnn.json'



### Error 2, [Traceback at line 50](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L50)<br />50..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.11/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/ztest.py", line 642, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/ztest.py", line 509, in test_all
<br />    log_remote_push()
<br />  File "https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71/mlmodels/ztest.py", line 154, in log_remote_push
<br />    tag = "m_" + str(arg.name)
<br />AttributeError: 'NoneType' object has no attribute 'name'

## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py<br />SyntaxError: invalid syntax



### Error 1, [Traceback at line 43](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L43)<br />43..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest.py", line 655, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest.py", line 509, in test_all
<br />    log_remote_push()
<br />  File "https://github.com/arita37/mlmodels/tree/e8ed27623b3d1702165de43d885398f64d2c399e/mlmodels/ztest.py", line 154, in log_remote_push
<br />    tag = "m_" + str(arg.name)
<br />AttributeError: 'NoneType' object has no attribute 'name'

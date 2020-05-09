## Original File URL: https://github.com/suyogdahal/mlmodels_store/blob/master/log_testall/log_testall_2020-05-06_00:14:00,380.txt


### Error 1, [Traceback at line 7](https://github.com/suyogdahal/mlmodels_store/blob/master/log_testall/log_testall_2020-05-06_00:14:00,380.txt#L7)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 424, in main
<br />    globals()[arg.do](arg)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/ztest.py", line 304, in test_all
<br />    cfg = json.load(open( path_norm(arg.config_file), mode='r'))['test_all']
<br />FileNotFoundError: [Errno 2] No such file or directory: 'config/test_config.json'

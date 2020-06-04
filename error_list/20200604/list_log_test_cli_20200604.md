## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 28](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L28)<br />28..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/ztest.py", line 638, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/ztest.py", line 363, in test_cli
<br />    with open( fileconfig, mode="r" ) as f:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/../README_usage_CLI.md'

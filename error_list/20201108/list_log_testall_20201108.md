## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 40](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L40)<br />40..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4cdc002cba85d4437aab96db0de9f52c658a62a5/mlmodels/model_keras//namentity_crm_bilstm.py", line 32, in <module>
<br />    from mlmodels.dataloader import DataLoader
<br />  File "https://github.com/arita37/mlmodels/tree/4cdc002cba85d4437aab96db0de9f52c658a62a5/mlmodels/dataloader.py", line 318
<br />    else :
<br />         ^
<br />IndentationError: unindent does not match any outer indentation level



### Error 2, [Traceback at line 47](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L47)<br />47..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/4cdc002cba85d4437aab96db0de9f52c658a62a5/mlmodels/ztest.py", line 655, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4cdc002cba85d4437aab96db0de9f52c658a62a5/mlmodels/ztest.py", line 509, in test_all
<br />    log_remote_push()
<br />  File "https://github.com/arita37/mlmodels/tree/4cdc002cba85d4437aab96db0de9f52c658a62a5/mlmodels/ztest.py", line 154, in log_remote_push
<br />    tag = "m_" + str(arg.name)
<br />AttributeError: 'NoneType' object has no attribute 'name'

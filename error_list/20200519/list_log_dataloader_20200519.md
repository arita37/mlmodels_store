## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 577](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L577)<br />577..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/516dda05a542721fda6d17f03e4351368ef767f3/mlmodels/dataloader.py", line 480, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/516dda05a542721fda6d17f03e4351368ef767f3/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 581](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L581)<br />581..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/516dda05a542721fda6d17f03e4351368ef767f3/mlmodels/dataloader.py", line 480, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/516dda05a542721fda6d17f03e4351368ef767f3/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 585](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L585)<br />585..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/516dda05a542721fda6d17f03e4351368ef767f3/mlmodels/dataloader.py", line 492, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/516dda05a542721fda6d17f03e4351368ef767f3/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

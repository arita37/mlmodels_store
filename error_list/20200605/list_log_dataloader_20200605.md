## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 541](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L541)<br />541..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataloader.py", line 297, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataloader.py", line 92, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

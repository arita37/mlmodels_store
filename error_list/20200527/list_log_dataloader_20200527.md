## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 537](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L537)<br />537..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/a497d63f77051de8adc79ecd064bbf7f55ecb632/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/a497d63f77051de8adc79ecd064bbf7f55ecb632/mlmodels/dataloader.py", line 297, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/a497d63f77051de8adc79ecd064bbf7f55ecb632/mlmodels/dataloader.py", line 92, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

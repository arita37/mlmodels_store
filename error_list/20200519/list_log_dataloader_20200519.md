## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 551](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L551)<br />551..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataloader.py", line 480, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 555](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L555)<br />555..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataloader.py", line 480, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 559](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L559)<br />559..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataloader.py", line 492, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 4, [Traceback at line 567](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L567)<br />567..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataloader.py", line 392, in test_run_model
<br />    print2(config)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pprint.py", line 121, in __init__
<br />    indent = int(indent)
<br />TypeError: int() argument must be a string, a bytes-like object or a number, not 'dict'



### Error 5, [Traceback at line 573](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L573)<br />573..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82c5b161058b5f6188d7af97c2913b706460ca41/mlmodels/dataloader.py", line 392, in test_run_model
<br />    print2(config)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pprint.py", line 121, in __init__
<br />    indent = int(indent)
<br />TypeError: int() argument must be a string, a bytes-like object or a number, not 'dict'

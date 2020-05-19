## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 552](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L552)<br />552..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 479, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 556](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L556)<br />556..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 479, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 560](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L560)<br />560..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 491, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 4, [Traceback at line 568](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L568)<br />568..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 391, in test_run_model
<br />    print2(config)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pprint.py", line 121, in __init__
<br />    indent = int(indent)
<br />TypeError: int() argument must be a string, a bytes-like object or a number, not 'dict'



### Error 5, [Traceback at line 574](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L574)<br />574..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 391, in test_run_model
<br />    print2(config)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pprint.py", line 121, in __init__
<br />    indent = int(indent)
<br />TypeError: int() argument must be a string, a bytes-like object or a number, not 'dict'



### Error 6, [Traceback at line 580](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L580)<br />580..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/07a220f833a9ced594f1652cb287e318217b235e/mlmodels/dataloader.py", line 391, in test_run_model
<br />    print2(config)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/pprint.py", line 121, in __init__
<br />    indent = int(indent)
<br />TypeError: int() argument must be a string, a bytes-like object or a number, not 'dict'

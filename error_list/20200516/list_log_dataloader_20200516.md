## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 669](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L669)<br />669..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 673](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L673)<br />673..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 677](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L677)<br />677..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 475, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 4, [Traceback at line 686](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L686)<br />686..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 376, in test_run_model
<br />    test_module( x['model_uri'],  path_norm_dict(x['pars']))
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/model_tch/torchhub.py", line 190, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'



### Error 5, [Traceback at line 1363](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1363)<br />1363..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 6, [Traceback at line 1367](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1367)<br />1367..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 7, [Traceback at line 1371](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1371)<br />1371..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 475, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 8, [Traceback at line 1380](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1380)<br />1380..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 376, in test_run_model
<br />    test_module( x['model_uri'],  path_norm_dict(x['pars']))
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/model_tch/torchhub.py", line 190, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'



### Error 9, [Traceback at line 2054](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2054)<br />2054..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 10, [Traceback at line 2058](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2058)<br />2058..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 11, [Traceback at line 2062](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2062)<br />2062..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 475, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 12, [Traceback at line 2071](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2071)<br />2071..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/dataloader.py", line 376, in test_run_model
<br />    test_module( x['model_uri'],  path_norm_dict(x['pars']))
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/5438088b9498570a1ef7df8d17148c905934a80a/mlmodels/model_tch/torchhub.py", line 214, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'

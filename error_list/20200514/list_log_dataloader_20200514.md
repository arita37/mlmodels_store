## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-14-09-25_0ad6d85c07617bb79b8698f0f5b9434fbf240370.py


### Error 1, [Traceback at line 488](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-14-09-25_0ad6d85c07617bb79b8698f0f5b9434fbf240370.py#L488)<br />488..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataloader.py", line 472, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 492](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-14-09-25_0ad6d85c07617bb79b8698f0f5b9434fbf240370.py#L492)<br />492..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataloader.py", line 472, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 496](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-14-09-25_0ad6d85c07617bb79b8698f0f5b9434fbf240370.py#L496)<br />496..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataloader.py", line 484, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 4, [Traceback at line 505](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-14-09-25_0ad6d85c07617bb79b8698f0f5b9434fbf240370.py#L505)<br />505..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/dataloader.py", line 376, in test_run_model
<br />    test_module( x['model_uri'],  path_norm_dict(x['pars']))
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/0ad6d85c07617bb79b8698f0f5b9434fbf240370/mlmodels/model_tch/torchhub.py", line 190, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'

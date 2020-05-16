## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-16-00-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py


### Error 1, [Traceback at line 490](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-16-00-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py#L490)<br />490..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 494](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-16-00-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py#L494)<br />494..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 498](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-16-00-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py#L498)<br />498..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataloader.py", line 475, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 4, [Traceback at line 507](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_2020-05-16-00-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py#L507)<br />507..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/dataloader.py", line 376, in test_run_model
<br />    test_module( x['model_uri'],  path_norm_dict(x['pars']))
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/18ac2a1774ee5f44eee229f2e2cad9101f46304e/mlmodels/model_tch/torchhub.py", line 190, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'

## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 487](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L487)<br />487..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json'



### Error 2, [Traceback at line 491](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L491)<br />491..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataloader.py", line 463, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 3, [Traceback at line 495](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L495)<br />495..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataloader.py", line 475, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataloader.py", line 326, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 4, [Traceback at line 504](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L504)<br />504..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/dataloader.py", line 376, in test_run_model
<br />    test_module( x['model_uri'],  path_norm_dict(x['pars']))
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd3fba21265c14a1d397be5ae9ee0a76c0c10804/mlmodels/model_tch/torchhub.py", line 214, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'

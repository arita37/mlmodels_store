## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 664](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L664)<br />664..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataloader.py", line 485, in test_dataloader
<br />    d = json.loads(open( f ).read())
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/json/refactor/model_list_CIFAR.json'



### Error 2, [Traceback at line 668](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L668)<br />668..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataloader.py", line 497, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataloader.py", line 324, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 78, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 3, [Traceback at line 1279](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1279)<br />1279..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataloader.py", line 398, in test_run_model
<br />    test_module(config['test']['model_pars']['model_uri'], param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras/dataloader/textcnn.py", line 77, in fit
<br />    validation_data=(Xtest, ytest))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training.py", line 1154, in fit
<br />    batch_size=batch_size)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training.py", line 579, in _standardize_user_data
<br />    exception_prefix='input')
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training_utils.py", line 145, in standardize_input_data
<br />    str(data_shape))
<br />ValueError: Error when checking input: expected input_1 to have shape (40,) but got array with shape (1,)



### Error 4, [Traceback at line 1294](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1294)<br />1294..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataloader.py", line 398, in test_run_model
<br />    test_module(config['test']['model_pars']['model_uri'], param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 264, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_tch/torchhub.py", line 214, in fit
<br />    model0.to(device)
<br />AttributeError: 'ProgressiveGAN' object has no attribute 'to'

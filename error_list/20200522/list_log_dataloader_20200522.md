## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 2401](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2401)<br />2401..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataloader.py", line 503, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataloader.py", line 324, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "mlmodels/dataloader.py", line 78, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 2, [Traceback at line 3323](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L3323)<br />3323..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataloader.py", line 403, in test_run_model
<br />    test_module(config['test']['model_pars']['model_uri'], param_pars, fittable = False if x in not_fittable_models else True)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/models.py", line 266, in test_module
<br />    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_keras/dataloader/textcnn.py", line 77, in fit
<br />    validation_data=(Xtest, ytest))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training.py", line 1154, in fit
<br />    batch_size=batch_size)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training.py", line 579, in _standardize_user_data
<br />    exception_prefix='input')
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training_utils.py", line 145, in standardize_input_data
<br />    str(data_shape))
<br />ValueError: Error when checking input: expected input_1 to have shape (40,) but got array with shape (1,)

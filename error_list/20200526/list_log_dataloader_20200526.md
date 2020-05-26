## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 470](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L470)<br />470..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/dataloader.py", line 500, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/dataloader.py", line 322, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/dataloader.py", line 75, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 2, [Traceback at line 4655](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L4655)<br />4655..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/dataloader.py", line 392, in test_run_model
<br />    test_module(config['test']['model_pars']['model_uri'], param_pars, fittable = False if x in not_fittable_models else True)
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/models.py", line 260, in test_module
<br />    model = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/model_keras/namentity_crm_bilstm.py", line 183, in get_dataset
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/dataloader.py", line 322, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/efc0e3fdb2ef28f9272e1435d71c61f048b27c99/mlmodels/dataloader.py", line 75, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

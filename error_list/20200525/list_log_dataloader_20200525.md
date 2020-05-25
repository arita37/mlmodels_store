## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 529](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L529)<br />529..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/dataloader.py", line 506, in test_dataloader
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/dataloader.py", line 327, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 2, [Traceback at line 1272](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L1272)<br />1272..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/dataloader.py", line 408, in test_run_model
<br />    test_module(config['test']['model_pars']['model_uri'], param_pars, fittable = False if x in not_fittable_models else True)
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/models.py", line 260, in test_module
<br />    model = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/model_keras/namentity_crm_bilstm.py", line 183, in get_dataset
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/dataloader.py", line 327, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/62bdf6db8ad57c05ac9066eac348673bac127d5b/mlmodels/dataloader.py", line 80, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

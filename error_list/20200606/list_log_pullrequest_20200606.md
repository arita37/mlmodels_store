## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-05-29-17-11_cd0e1dbcf68c34dccf0d76405c260752e880d933.py


### Error 1, [Traceback at line 337](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-05-29-17-11_cd0e1dbcf68c34dccf0d76405c260752e880d933.py#L337)<br />337..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/textcnn.py", line 258, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/textcnn.py", line 165, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 2, [Traceback at line 345](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-05-29-17-11_cd0e1dbcf68c34dccf0d76405c260752e880d933.py#L345)<br />345..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest.py", line 640, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest.py", line 426, in test_pullrequest
<br />    raise Exception(f"Unknown dataset type", x)
<br />Exception: ('Unknown dataset type', "FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/json/refactor/textcnn_keras.json'\n")

## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-07-14-06-19_077ac9573b3255f0836baba55f19fb6dbaa40c9d.py


### Error 1, [Traceback at line 339](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-07-14-06-19_077ac9573b3255f0836baba55f19fb6dbaa40c9d.py#L339)<br />339..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/model_keras/textcnn.py", line 258, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/model_keras/textcnn.py", line 165, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 2, [Traceback at line 351](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-07-14-06-19_077ac9573b3255f0836baba55f19fb6dbaa40c9d.py#L351)<br />351..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/ztest.py", line 642, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/ztest.py", line 424, in test_pullrequest
<br />    raise Exception(f"Unknown dataset type", x)
<br />Exception: ('Unknown dataset type', "FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d/mlmodels/dataset/json/refactor/textcnn_keras.json'\n")

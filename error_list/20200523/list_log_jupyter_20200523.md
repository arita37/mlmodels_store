## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 45](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L45)<br />45..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 2, [Traceback at line 57](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L57)<br />57..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 64](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L64)<br />64..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/example//sklearn_titanic_svm.py", line 20, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 4, [Traceback at line 80](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L80)<br />80..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/example//lightgbm.py", line 9, in <module>
<br />    get_ipython().run_line_magic('load_ext', 'autoreload')
<br />NameError: name 'get_ipython' is not defined



### Error 5, [Traceback at line 94](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L94)<br />94..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/example//sklearn_titanic_randomForest.py", line 7, in <module>
<br />    get_ipython().run_line_magic('load_ext', 'autoreload')
<br />NameError: name 'get_ipython' is not defined



### Error 6, [Traceback at line 108](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L108)<br />108..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/ztest.py", line 589, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/ztest.py", line 252, in test_jupyter
<br />    os_file_replace(file2, s1="%", s2="# %")   #### Jupyter Flag
<br />  File "https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/ztest.py", line 258, in os_file_replace
<br />    with open(filename, 'r') as file :
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/9dc6ff71662734477d97e9bd10a210ba608dd19d/mlmodels/example//timeseries_m5_deepar.py'

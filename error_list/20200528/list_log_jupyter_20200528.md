## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 45](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L45)<br />45..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 2, [Traceback at line 57](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L57)<br />57..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 64](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L64)<br />64..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//sklearn_titanic_svm.py", line 20, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 4, [Traceback at line 80](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L80)<br />80..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/model_sklearn/model_lightgbm.py", line 68, in <module>
<br />    from lightgbm import LGBMModel
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example/lightgbm.py", line 27, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'lightgbm_titanic.json'



### Error 5, [Traceback at line 99](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L99)<br />99..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 106](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L106)<br />106..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//lightgbm.py", line 23, in <module>
<br />    module        =  module_load( model_uri= model_uri)
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.model_lightgbm notfound, [Errno 2] No such file or directory: 'lightgbm_titanic.json', tuple index out of range



### Error 7, [Traceback at line 122](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L122)<br />122..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 8, [Traceback at line 134](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L134)<br />134..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 141](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L141)<br />141..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//sklearn_titanic_randomForest.py", line 21, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 10, [Traceback at line 182](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L182)<br />182..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//lightgbm_home_retail.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'



### Error 11, [Traceback at line 196](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L196)<br />196..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//keras_charcnn_reuters.py", line 28, in <module>
<br />    pars = json.load(open( config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'reuters_charcnn.json'



### Error 12, [Traceback at line 255](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L255)<br />255..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/model_sklearn/model_lightgbm.py", line 68, in <module>
<br />    from lightgbm import LGBMModel
<br />ImportError: cannot import name 'LGBMModel'



### Error 13, [Traceback at line 272](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L272)<br />272..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 14, [Traceback at line 279](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L279)<br />279..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
<br />    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 258, in hyperparameter_tune
<br />    dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, params=self.params, X_test=X_test, Y_test=Y_test)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 204, in generate_datasets
<br />    dataset_train = construct_dataset(x=X_train, y=Y_train, location=self.path + 'datasets/train', params=data_params, save=save, weight=W_train)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/utils.py", line 52, in construct_dataset
<br />    try_import_lightgbm()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/try_import.py", line 13, in try_import_lightgbm
<br />    import lightgbm
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example/lightgbm.py", line 23, in <module>
<br />    module        =  module_load( model_uri= model_uri)
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.model_lightgbm notfound, cannot import name 'LGBMModel', tuple index out of range
<br />SyntaxError: invalid syntax



### Error 15, [Traceback at line 505](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L505)<br />505..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 16, [Traceback at line 554](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L554)<br />554..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//keras-textcnn.py", line 37, in <module>
<br />    _, _   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 17, [Traceback at line 573](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L573)<br />573..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//sklearn_titanic_randomForest_example2.py", line 22, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'



### Error 18, [Traceback at line 601](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L601)<br />601..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//gluon_automl_titanic.py", line 27, in <module>
<br />    data_path= '../mlmodels/dataset/json/gluon_automl.json'
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/model_gluon/gluon_automl.py", line 82, in get_params
<br />    with open(data_path, encoding='utf-8') as config_f:
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'



### Error 19, [Traceback at line 617](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L617)<br />617..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//tensorflow__lstm_json.py", line 13, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 20, [Traceback at line 631](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L631)<br />631..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 21, [Traceback at line 643](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L643)<br />643..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 22, [Traceback at line 650](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L650)<br />650..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 23, [Traceback at line 667](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L667)<br />667..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//lightgbm_titanic.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'
<br />SyntaxError: invalid syntax



### Error 24, [Traceback at line 699](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L699)<br />699..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//arun_hyper.py", line 2, in <module>
<br />    from jsoncomment import JsonComment ; json = JsonComment(), copy
<br />NameError: name 'copy' is not defined



### Error 25, [Traceback at line 711](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L711)<br />711..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 26, [Traceback at line 723](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L723)<br />723..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 27, [Traceback at line 735](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L735)<br />735..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 28, [Traceback at line 742](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L742)<br />742..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//benchmark_timeseries_m5.py", line 27, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 29, [Traceback at line 784](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L784)<br />784..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example//arun_model.py", line 27, in <module>
<br />    pars = json.load(open(config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/model_keras/ardmn.json'



### Error 30, [Traceback at line 804](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L804)<br />804..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 31, [Traceback at line 816](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L816)<br />816..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 32, [Traceback at line 823](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L823)<br />823..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example/benchmark_timeseries_m5.py", line 27, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/097d79840b302a3e2f9efdf341ce6b4d32d7a46c/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range

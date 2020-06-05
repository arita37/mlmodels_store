## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 45](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L45)<br />45..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 2, [Traceback at line 57](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L57)<br />57..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 64](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L64)<br />64..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//sklearn_titanic_svm.py", line 20, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 4, [Traceback at line 80](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L80)<br />80..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/model_sklearn/model_lightgbm.py", line 316
<br />    else:
<br />       ^
<br />SyntaxError: invalid syntax



### Error 5, [Traceback at line 100](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L100)<br />100..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 107](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L107)<br />107..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//lightgbm.py", line 23, in <module>
<br />    module        =  module_load( model_uri= model_uri)
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.model_lightgbm notfound, invalid syntax (model_lightgbm.py, line 316), tuple index out of range



### Error 7, [Traceback at line 123](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L123)<br />123..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 8, [Traceback at line 135](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L135)<br />135..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 142](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L142)<br />142..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//sklearn_titanic_randomForest.py", line 21, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 10, [Traceback at line 183](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L183)<br />183..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//lightgbm_home_retail.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'



### Error 11, [Traceback at line 197](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L197)<br />197..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//keras_charcnn_reuters.py", line 28, in <module>
<br />    pars = json.load(open( config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'reuters_charcnn.json'



### Error 12, [Traceback at line 256](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L256)<br />256..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/model_sklearn/model_lightgbm.py", line 316
<br />    else:
<br />       ^
<br />SyntaxError: invalid syntax



### Error 13, [Traceback at line 276](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L276)<br />276..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 14, [Traceback at line 283](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L283)<br />283..Traceback (most recent call last):
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
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example/lightgbm.py", line 23, in <module>
<br />    module        =  module_load( model_uri= model_uri)
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.model_lightgbm notfound, invalid syntax (model_lightgbm.py, line 316), tuple index out of range
<br />SyntaxError: invalid syntax



### Error 15, [Traceback at line 567](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L567)<br />567..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 16, [Traceback at line 616](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L616)<br />616..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//keras-textcnn.py", line 37, in <module>
<br />    _, _   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 17, [Traceback at line 635](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L635)<br />635..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//sklearn_titanic_randomForest_example2.py", line 22, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'



### Error 18, [Traceback at line 663](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L663)<br />663..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//gluon_automl_titanic.py", line 27, in <module>
<br />    data_path= '../mlmodels/dataset/json/gluon_automl.json'
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/model_gluon/gluon_automl.py", line 82, in get_params
<br />    with open(data_path, encoding='utf-8') as config_f:
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'



### Error 19, [Traceback at line 679](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L679)<br />679..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//tensorflow__lstm_json.py", line 13, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 20, [Traceback at line 693](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L693)<br />693..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 21, [Traceback at line 705](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L705)<br />705..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 22, [Traceback at line 712](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L712)<br />712..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 23, [Traceback at line 729](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L729)<br />729..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//lightgbm_titanic.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'
<br />SyntaxError: invalid syntax



### Error 24, [Traceback at line 761](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L761)<br />761..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//arun_hyper.py", line 2, in <module>
<br />    from jsoncomment import JsonComment ; json = JsonComment(), copy
<br />NameError: name 'copy' is not defined



### Error 25, [Traceback at line 773](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L773)<br />773..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 26, [Traceback at line 785](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L785)<br />785..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 27, [Traceback at line 797](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L797)<br />797..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 28, [Traceback at line 804](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L804)<br />804..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//benchmark_timeseries_m5.py", line 27, in <module>
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
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 29, [Traceback at line 846](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L846)<br />846..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example//arun_model.py", line 27, in <module>
<br />    pars = json.load(open(config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/model_keras/ardmn.json'



### Error 30, [Traceback at line 866](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L866)<br />866..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 31, [Traceback at line 878](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L878)<br />878..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 32, [Traceback at line 885](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L885)<br />885..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example/benchmark_timeseries_m5.py", line 27, in <module>
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
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/d67c613373df800885b9f1e6941d6f5879aa2c04/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range

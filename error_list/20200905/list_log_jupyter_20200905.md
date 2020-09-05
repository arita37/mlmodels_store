## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 45](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L45)<br />45..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 2, [Traceback at line 57](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L57)<br />57..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 64](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L64)<br />64..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//sklearn_titanic_randomForest.py", line 21, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 4, [Traceback at line 81](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L81)<br />81..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//sklearn_titanic_randomForest_example2.py", line 22, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'



### Error 5, [Traceback at line 109](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L109)<br />109..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//gluon_automl_titanic.py", line 27, in <module>
<br />    data_path= '../mlmodels/dataset/json/gluon_automl.json'
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/model_gluon/gluon_automl.py", line 82, in get_params
<br />    with open(data_path, encoding='utf-8') as config_f:
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'



### Error 6, [Traceback at line 126](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L126)<br />126..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//lightgbm_titanic.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'



### Error 7, [Traceback at line 140](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L140)<br />140..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 8, [Traceback at line 152](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L152)<br />152..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 159](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L159)<br />159..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 10, [Traceback at line 175](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L175)<br />175..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 11, [Traceback at line 187](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L187)<br />187..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 12, [Traceback at line 194](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L194)<br />194..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//sklearn_titanic_svm.py", line 20, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 13, [Traceback at line 210](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L210)<br />210..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//tensorflow__lstm_json.py", line 13, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 14, [Traceback at line 224](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L224)<br />224..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 15, [Traceback at line 236](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L236)<br />236..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 16, [Traceback at line 243](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L243)<br />243..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//gluon_automl.py", line 9, in <module>
<br />    import autogluon as ag
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/autogluon/__init__.py", line 6, in <module>
<br />    from .utils.try_import import *
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/autogluon/utils/__init__.py", line 5, in <module>
<br />    from .dataset import *
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/autogluon/utils/dataset.py", line 3, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 17, [Traceback at line 293](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L293)<br />293..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 18, [Traceback at line 305](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L305)<br />305..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 19, [Traceback at line 312](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L312)<br />312..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/model_tf/1_lstm.py", line 13, in <module>
<br />    from sklearn.preprocessing import MinMaxScaler
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 20, [Traceback at line 333](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L333)<br />333..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 21, [Traceback at line 340](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L340)<br />340..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//tensorflow_1_lstm.py", line 47, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tf.1_lstm notfound, Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range, tuple index out of range
<br />SyntaxError: invalid syntax



### Error 22, [Traceback at line 371](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L371)<br />371..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//lightgbm_home_retail.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'



### Error 23, [Traceback at line 397](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L397)<br />397..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 24, [Traceback at line 411](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L411)<br />411..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/model_sklearn/model_lightgbm.py", line 316
<br />    else:
<br />       ^
<br />SyntaxError: invalid syntax



### Error 25, [Traceback at line 431](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L431)<br />431..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 26, [Traceback at line 438](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L438)<br />438..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//lightgbm.py", line 23, in <module>
<br />    module        =  module_load( model_uri= model_uri)
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.model_lightgbm notfound, invalid syntax (model_lightgbm.py, line 316), tuple index out of range



### Error 27, [Traceback at line 455](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L455)<br />455..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 28, [Traceback at line 467](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L467)<br />467..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 29, [Traceback at line 474](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L474)<br />474..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/model_keras/textcnn.py", line 31, in <module>
<br />    from mlmodels.dataloader import DataLoader
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/dataloader.py", line 69, in <module>
<br />    from sklearn.model_selection import train_test_split
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 30, [Traceback at line 497](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L497)<br />497..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 31, [Traceback at line 504](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L504)<br />504..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//keras-textcnn.py", line 35, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.textcnn notfound, Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range, tuple index out of range



### Error 32, [Traceback at line 544](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L544)<br />544..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//keras_charcnn_reuters.py", line 28, in <module>
<br />    pars = json.load(open( config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'reuters_charcnn.json'



### Error 33, [Traceback at line 556](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L556)<br />556..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 34, [Traceback at line 568](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L568)<br />568..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 35, [Traceback at line 575](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L575)<br />575..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example/benchmark_timeseries_m5.py", line 27, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 36, [Traceback at line 623](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L623)<br />623..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 37, [Traceback at line 635](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L635)<br />635..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 38, [Traceback at line 642](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L642)<br />642..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//benchmark_timeseries_m5.py", line 27, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />SyntaxError: invalid syntax



### Error 39, [Traceback at line 696](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L696)<br />696..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//arun_model.py", line 27, in <module>
<br />    pars = json.load(open(config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/model_keras/ardmn.json'



### Error 40, [Traceback at line 716](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L716)<br />716..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 41, [Traceback at line 728](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L728)<br />728..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/ff21906d63b46d89670bd3ed148b01355ca5ff0e/mlmodels/example//arun_hyper.py", line 2, in <module>
<br />    from jsoncomment import JsonComment ; json = JsonComment(), copy
<br />NameError: name 'copy' is not defined


  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '0635d2a358ad260f77f69ce3b3238ee806f53e4b', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/0635d2a358ad260f77f69ce3b3238ee806f53e4b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files #########################################
['/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//timeseries_m5_deepar.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py', '/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py']





 ************************************************************************************************************************
############ Running Jupyter files ################################





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_svm.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb to script
[NbConvertApp] Writing 1274 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/sklearn_titanic_svm.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.py", line 20, in <module>
    module        =  module_load( model_uri= model_uri )                           # Load file definition
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb to script
[NbConvertApp] Writing 1191 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py", line 68, in <module>
    from lightgbm import LGBMModel
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm.py", line 27, in <module>
    pars = json.load(open( data_path , mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'lightgbm_titanic.json'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.py", line 23, in <module>
    module        =  module_load( model_uri= model_uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.model_lightgbm notfound, [Errno 2] No such file or directory: 'lightgbm_titanic.json', tuple index out of range





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb to script
[NbConvertApp] Writing 1469 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/sklearn_titanic_randomForest.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.py", line 21, in <module>
    module        =  module_load( model_uri= model_uri )                           # Load file definition
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//timeseries_m5_deepar.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//timeseries_m5_deepar.ipynb to script
[NbConvertApp] Writing 16958 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/timeseries_m5_deepar.txt
python: can't open file '/home/runner/work/mlmodels/mlmodels/mlmodels/example//timeseries_m5_deepar.py': [Errno 2] No such file or directory
No replacement





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//fashion_MNIST_mlmodels.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb to script
[NbConvertApp] Writing 1882 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/fashion_MNIST_mlmodels.txt
python: can't open file '/home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.py': [Errno 2] No such file or directory
No replacement





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_home_retail.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb to script
[NbConvertApp] Writing 1357 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm_home_retail.py
Deprecaton set to False
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.py", line 21, in <module>
    pars = json.load(open( data_path , mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras_charcnn_reuters.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb to script
[NbConvertApp] Writing 1067 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/keras_charcnn_reuters.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.py", line 28, in <module>
    pars = json.load(open( config_path , mode='r'))[config_mode]
FileNotFoundError: [Errno 2] No such file or directory: 'reuters_charcnn.json'





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb to script
[NbConvertApp] Writing 1819 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/gluon_automl.py
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073
Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.
Beginning AutoGluon training ... Time limit = 120s
AutoGluon will save models to dataset/
Train Data Rows:    39073
Train Data Columns: 15
Preprocessing data ...
Here are the first 10 unique label values in your data:  [' Tech-support' ' Transport-moving' ' Other-service' ' ?'
 ' Handlers-cleaners' ' Sales' ' Craft-repair' ' Adm-clerical'
 ' Exec-managerial' ' Prof-specialty']
AutoGluon infers your prediction problem is: multiclass  (because dtype of label-column == object)
If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])

Feature Generator processed 39073 data points with 14 features
Original Features:
	int features: 6
	object features: 8
Generated Features:
	int features: 0
All Features:
	int features: 6
	object features: 8
	Data preprocessing and feature engineering runtime = 0.24s ...
AutoGluon will gauge predictive performance using evaluation metric: accuracy
To change this, specify the eval_metric argument of fit()
AutoGluon will early stop models using evaluation metric: accuracy
Saving dataset/learner.pkl
Beginning hyperparameter tuning for Gradient Boosting Model...
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 258, in hyperparameter_tune
    dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, params=self.params, X_test=X_test, Y_test=Y_test)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 204, in generate_datasets
    dataset_train = construct_dataset(x=X_train, y=Y_train, location=self.path + 'datasets/train', params=data_params, save=save, weight=W_train)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/utils.py", line 52, in construct_dataset
    try_import_lightgbm()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/try_import.py", line 13, in try_import_lightgbm
    import lightgbm
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm.py", line 23, in <module>
    module        =  module_load( model_uri= model_uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
Warning: Exception caused LightGBMClassifier to fail during hyperparameter tuning... Skipping this model.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py", line 68, in <module>
    from lightgbm import LGBMModel
ImportError: cannot import name 'LGBMModel'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 258, in hyperparameter_tune
    dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, params=self.params, X_test=X_test, Y_test=Y_test)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 204, in generate_datasets
    dataset_train = construct_dataset(x=X_train, y=Y_train, location=self.path + 'datasets/train', params=data_params, save=save, weight=W_train)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/utils.py", line 52, in construct_dataset
    try_import_lightgbm()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/try_import.py", line 13, in try_import_lightgbm
    import lightgbm
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm.py", line 23, in <module>
    module        =  module_load( model_uri= model_uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.model_lightgbm notfound, cannot import name 'LGBMModel', tuple index out of range
Saving dataset/models/trainer.pkl
Beginning hyperparameter tuning for Neural Network...
Hyperparameter search space for Neural Network: 
network_type:   Categorical['widedeep', 'feedforward']
layers:   Categorical[[100], [1000], [200, 100], [300, 200, 100]]
activation:   Categorical['relu', 'softrelu', 'tanh']
embedding_size_factor:   Real: lower=0.5, upper=1.5
use_batchnorm:   Categorical[True, False]
dropout_prob:   Real: lower=0.0, upper=0.5
learning_rate:   Real: lower=0.0001, upper=0.01
weight_decay:   Real: lower=1e-12, upper=0.1
AutoGluon Neural Network infers features are of the following types:
{
    "continuous": [
        "age",
        "education-num",
        "hours-per-week"
    ],
    "skewed": [
        "fnlwgt",
        "capital-gain",
        "capital-loss"
    ],
    "onehot": [
        "sex",
        "class"
    ],
    "embed": [
        "workclass",
        "education",
        "marital-status",
        "relationship",
        "race",
        "native-country"
    ],
    "language": []
}


Saving dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:48<01:13, 24.48s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.12597819303545657, 'embedding_size_factor': 1.4324666034296287, 'layers.choice': 0, 'learning_rate': 0.004368167916432742, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.455521440735789e-09} and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc0 \r\xad\x8cX\xd5X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\xebb\x19\xe5z\xb4X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?q\xe4[%\x83"AX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>%\x17\xbf\xaeA\x84\xf2u.' and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc0 \r\xad\x8cX\xd5X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\xebb\x19\xe5z\xb4X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?q\xe4[%\x83"AX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>%\x17\xbf\xaeA\x84\xf2u.' and reward: 0.3906
 60%|██████    | 3/5 [01:38<01:04, 32.04s/it] 60%|██████    | 3/5 [01:38<01:05, 32.88s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.4434918833669035, 'embedding_size_factor': 1.3131805922764617, 'layers.choice': 2, 'learning_rate': 0.0007696250322110364, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.002041125061557475} and reward: 0.3852
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdcb+\xc7\xc6\x8c\x08X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x02\xc9\xa7\x19\x1b\xe8X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?I8\x15+\xf8\xc5\tX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?`\xb8\x8c\xac]\xad\xa6u.' and reward: 0.3852
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdcb+\xc7\xc6\x8c\x08X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x02\xc9\xa7\x19\x1b\xe8X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?I8\x15+\xf8\xc5\tX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?`\xb8\x8c\xac]\xad\xa6u.' and reward: 0.3852
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 149.1022334098816
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.12597819303545657, 'embedding_size_factor': 1.4324666034296287, 'layers.choice': 0, 'learning_rate': 0.004368167916432742, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.455521440735789e-09}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -31.74s of remaining time.
Ensemble size: 84
Ensemble weights: 
[0.54761905 0.17857143 0.27380952]
	0.3952	 = Validation accuracy score
	1.0s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 152.78s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vison_fashion_MNIST.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb to script
[NbConvertApp] Writing 1882 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/vison_fashion_MNIST.txt
python: can't open file '/home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.py': [Errno 2] No such file or directory
No replacement





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow_1_lstm.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb to script
[NbConvertApp] Writing 1829 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/tensorflow_1_lstm.py
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
/home/runner/work/mlmodels/mlmodels
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb to script
[NbConvertApp] Writing 7241 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/vision_mnist.txt
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py", line 15
    !git clone https://github.com/ahmed3bbas/mlmodels.git
    ^
SyntaxError: invalid syntax





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb to script
[NbConvertApp] Writing 1566 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm_glass.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py", line 16, in <module>
    print( os.getcwd())
NameError: name 'os' is not defined





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras-textcnn.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb to script
[NbConvertApp] Writing 1251 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/keras-textcnn.py
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 400)          0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 400, 50)      500         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 398, 128)     19328       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 397, 128)     25728       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 396, 128)     32128       embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 78,069
Trainable params: 78,069
Non-trainable params: 0
__________________________________________________________________________________________________
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3497984/17464789 [=====>........................] - ETA: 0s
 9781248/17464789 [===============>..............] - ETA: 0s
17178624/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-27 15:20:23.885058: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-27 15:20:23.888979: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-27 15:20:23.889120: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e1ca856440 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 15:20:23.889133: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:24 - loss: 8.6249 - accuracy: 0.4375
   64/25000 [..............................] - ETA: 2:44 - loss: 8.3854 - accuracy: 0.4531
   96/25000 [..............................] - ETA: 2:08 - loss: 8.3055 - accuracy: 0.4583
  128/25000 [..............................] - ETA: 1:52 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:41 - loss: 7.8583 - accuracy: 0.4875
  192/25000 [..............................] - ETA: 1:34 - loss: 8.2256 - accuracy: 0.4635
  224/25000 [..............................] - ETA: 1:29 - loss: 8.0773 - accuracy: 0.4732
  256/25000 [..............................] - ETA: 1:25 - loss: 7.5468 - accuracy: 0.5078
  288/25000 [..............................] - ETA: 1:22 - loss: 7.4537 - accuracy: 0.5139
  320/25000 [..............................] - ETA: 1:20 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:18 - loss: 7.6231 - accuracy: 0.5028
  384/25000 [..............................] - ETA: 1:16 - loss: 7.5868 - accuracy: 0.5052
  416/25000 [..............................] - ETA: 1:14 - loss: 7.6666 - accuracy: 0.5000
  448/25000 [..............................] - ETA: 1:14 - loss: 7.6666 - accuracy: 0.5000
  480/25000 [..............................] - ETA: 1:12 - loss: 7.5388 - accuracy: 0.5083
  512/25000 [..............................] - ETA: 1:11 - loss: 7.5768 - accuracy: 0.5059
  544/25000 [..............................] - ETA: 1:11 - loss: 7.5539 - accuracy: 0.5074
  576/25000 [..............................] - ETA: 1:10 - loss: 7.4803 - accuracy: 0.5122
  608/25000 [..............................] - ETA: 1:09 - loss: 7.3892 - accuracy: 0.5181
  640/25000 [..............................] - ETA: 1:08 - loss: 7.3552 - accuracy: 0.5203
  672/25000 [..............................] - ETA: 1:08 - loss: 7.4384 - accuracy: 0.5149
  704/25000 [..............................] - ETA: 1:07 - loss: 7.3835 - accuracy: 0.5185
  736/25000 [..............................] - ETA: 1:06 - loss: 7.3750 - accuracy: 0.5190
  768/25000 [..............................] - ETA: 1:06 - loss: 7.4470 - accuracy: 0.5143
  800/25000 [..............................] - ETA: 1:06 - loss: 7.4941 - accuracy: 0.5113
  832/25000 [..............................] - ETA: 1:05 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:05 - loss: 7.4004 - accuracy: 0.5174
  896/25000 [>.............................] - ETA: 1:04 - loss: 7.3757 - accuracy: 0.5190
  928/25000 [>.............................] - ETA: 1:04 - loss: 7.3692 - accuracy: 0.5194
  960/25000 [>.............................] - ETA: 1:04 - loss: 7.4270 - accuracy: 0.5156
  992/25000 [>.............................] - ETA: 1:04 - loss: 7.4966 - accuracy: 0.5111
 1024/25000 [>.............................] - ETA: 1:04 - loss: 7.5468 - accuracy: 0.5078
 1056/25000 [>.............................] - ETA: 1:03 - loss: 7.5505 - accuracy: 0.5076
 1088/25000 [>.............................] - ETA: 1:03 - loss: 7.5821 - accuracy: 0.5055
 1120/25000 [>.............................] - ETA: 1:03 - loss: 7.5708 - accuracy: 0.5063
 1152/25000 [>.............................] - ETA: 1:02 - loss: 7.5601 - accuracy: 0.5069
 1184/25000 [>.............................] - ETA: 1:02 - loss: 7.5889 - accuracy: 0.5051
 1216/25000 [>.............................] - ETA: 1:02 - loss: 7.6162 - accuracy: 0.5033
 1248/25000 [>.............................] - ETA: 1:01 - loss: 7.5438 - accuracy: 0.5080
 1280/25000 [>.............................] - ETA: 1:01 - loss: 7.5229 - accuracy: 0.5094
 1312/25000 [>.............................] - ETA: 1:01 - loss: 7.5498 - accuracy: 0.5076
 1344/25000 [>.............................] - ETA: 1:01 - loss: 7.5639 - accuracy: 0.5067
 1376/25000 [>.............................] - ETA: 1:00 - loss: 7.5552 - accuracy: 0.5073
 1408/25000 [>.............................] - ETA: 1:00 - loss: 7.5795 - accuracy: 0.5057
 1440/25000 [>.............................] - ETA: 1:00 - loss: 7.5814 - accuracy: 0.5056
 1472/25000 [>.............................] - ETA: 1:00 - loss: 7.5625 - accuracy: 0.5068
 1504/25000 [>.............................] - ETA: 1:00 - loss: 7.5443 - accuracy: 0.5080
 1536/25000 [>.............................] - ETA: 1:00 - loss: 7.5368 - accuracy: 0.5085
 1568/25000 [>.............................] - ETA: 59s - loss: 7.5493 - accuracy: 0.5077 
 1600/25000 [>.............................] - ETA: 59s - loss: 7.5420 - accuracy: 0.5081
 1632/25000 [>.............................] - ETA: 59s - loss: 7.5069 - accuracy: 0.5104
 1664/25000 [>.............................] - ETA: 59s - loss: 7.4915 - accuracy: 0.5114
 1696/25000 [=>............................] - ETA: 59s - loss: 7.5310 - accuracy: 0.5088
 1728/25000 [=>............................] - ETA: 58s - loss: 7.5335 - accuracy: 0.5087
 1760/25000 [=>............................] - ETA: 58s - loss: 7.5272 - accuracy: 0.5091
 1792/25000 [=>............................] - ETA: 58s - loss: 7.5383 - accuracy: 0.5084
 1824/25000 [=>............................] - ETA: 58s - loss: 7.5573 - accuracy: 0.5071
 1856/25000 [=>............................] - ETA: 58s - loss: 7.6005 - accuracy: 0.5043
 1888/25000 [=>............................] - ETA: 58s - loss: 7.6341 - accuracy: 0.5021
 1920/25000 [=>............................] - ETA: 58s - loss: 7.5708 - accuracy: 0.5063
 1952/25000 [=>............................] - ETA: 57s - loss: 7.5724 - accuracy: 0.5061
 1984/25000 [=>............................] - ETA: 57s - loss: 7.5893 - accuracy: 0.5050
 2016/25000 [=>............................] - ETA: 57s - loss: 7.5906 - accuracy: 0.5050
 2048/25000 [=>............................] - ETA: 57s - loss: 7.5992 - accuracy: 0.5044
 2080/25000 [=>............................] - ETA: 57s - loss: 7.6371 - accuracy: 0.5019
 2112/25000 [=>............................] - ETA: 57s - loss: 7.6303 - accuracy: 0.5024
 2144/25000 [=>............................] - ETA: 57s - loss: 7.6380 - accuracy: 0.5019
 2176/25000 [=>............................] - ETA: 57s - loss: 7.6243 - accuracy: 0.5028
 2208/25000 [=>............................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 2240/25000 [=>............................] - ETA: 57s - loss: 7.6529 - accuracy: 0.5009
 2272/25000 [=>............................] - ETA: 57s - loss: 7.6599 - accuracy: 0.5004
 2304/25000 [=>............................] - ETA: 57s - loss: 7.6799 - accuracy: 0.4991
 2336/25000 [=>............................] - ETA: 56s - loss: 7.7060 - accuracy: 0.4974
 2368/25000 [=>............................] - ETA: 56s - loss: 7.6731 - accuracy: 0.4996
 2400/25000 [=>............................] - ETA: 56s - loss: 7.6219 - accuracy: 0.5029
 2432/25000 [=>............................] - ETA: 56s - loss: 7.6288 - accuracy: 0.5025
 2464/25000 [=>............................] - ETA: 56s - loss: 7.6480 - accuracy: 0.5012
 2496/25000 [=>............................] - ETA: 56s - loss: 7.6482 - accuracy: 0.5012
 2528/25000 [==>...........................] - ETA: 56s - loss: 7.6424 - accuracy: 0.5016
 2560/25000 [==>...........................] - ETA: 56s - loss: 7.6367 - accuracy: 0.5020
 2592/25000 [==>...........................] - ETA: 56s - loss: 7.6252 - accuracy: 0.5027
 2624/25000 [==>...........................] - ETA: 55s - loss: 7.5848 - accuracy: 0.5053
 2656/25000 [==>...........................] - ETA: 55s - loss: 7.6031 - accuracy: 0.5041
 2688/25000 [==>...........................] - ETA: 55s - loss: 7.5925 - accuracy: 0.5048
 2720/25000 [==>...........................] - ETA: 55s - loss: 7.5764 - accuracy: 0.5059
 2752/25000 [==>...........................] - ETA: 55s - loss: 7.5775 - accuracy: 0.5058
 2784/25000 [==>...........................] - ETA: 55s - loss: 7.5950 - accuracy: 0.5047
 2816/25000 [==>...........................] - ETA: 55s - loss: 7.5904 - accuracy: 0.5050
 2848/25000 [==>...........................] - ETA: 55s - loss: 7.5751 - accuracy: 0.5060
 2880/25000 [==>...........................] - ETA: 54s - loss: 7.5708 - accuracy: 0.5063
 2912/25000 [==>...........................] - ETA: 54s - loss: 7.5824 - accuracy: 0.5055
 2944/25000 [==>...........................] - ETA: 54s - loss: 7.5989 - accuracy: 0.5044
 2976/25000 [==>...........................] - ETA: 54s - loss: 7.6099 - accuracy: 0.5037
 3008/25000 [==>...........................] - ETA: 54s - loss: 7.6054 - accuracy: 0.5040
 3040/25000 [==>...........................] - ETA: 54s - loss: 7.5960 - accuracy: 0.5046
 3072/25000 [==>...........................] - ETA: 54s - loss: 7.5918 - accuracy: 0.5049
 3104/25000 [==>...........................] - ETA: 54s - loss: 7.5777 - accuracy: 0.5058
 3136/25000 [==>...........................] - ETA: 54s - loss: 7.5542 - accuracy: 0.5073
 3168/25000 [==>...........................] - ETA: 53s - loss: 7.5505 - accuracy: 0.5076
 3200/25000 [==>...........................] - ETA: 53s - loss: 7.5564 - accuracy: 0.5072
 3232/25000 [==>...........................] - ETA: 53s - loss: 7.5622 - accuracy: 0.5068
 3264/25000 [==>...........................] - ETA: 53s - loss: 7.5445 - accuracy: 0.5080
 3296/25000 [==>...........................] - ETA: 53s - loss: 7.5457 - accuracy: 0.5079
 3328/25000 [==>...........................] - ETA: 53s - loss: 7.5560 - accuracy: 0.5072
 3360/25000 [===>..........................] - ETA: 53s - loss: 7.5754 - accuracy: 0.5060
 3392/25000 [===>..........................] - ETA: 53s - loss: 7.5898 - accuracy: 0.5050
 3424/25000 [===>..........................] - ETA: 53s - loss: 7.5905 - accuracy: 0.5050
 3456/25000 [===>..........................] - ETA: 52s - loss: 7.5823 - accuracy: 0.5055
 3488/25000 [===>..........................] - ETA: 52s - loss: 7.5787 - accuracy: 0.5057
 3520/25000 [===>..........................] - ETA: 52s - loss: 7.5795 - accuracy: 0.5057
 3552/25000 [===>..........................] - ETA: 52s - loss: 7.5803 - accuracy: 0.5056
 3584/25000 [===>..........................] - ETA: 52s - loss: 7.5811 - accuracy: 0.5056
 3616/25000 [===>..........................] - ETA: 52s - loss: 7.5776 - accuracy: 0.5058
 3648/25000 [===>..........................] - ETA: 52s - loss: 7.5868 - accuracy: 0.5052
 3680/25000 [===>..........................] - ETA: 52s - loss: 7.5875 - accuracy: 0.5052
 3712/25000 [===>..........................] - ETA: 52s - loss: 7.5881 - accuracy: 0.5051
 3744/25000 [===>..........................] - ETA: 52s - loss: 7.5765 - accuracy: 0.5059
 3776/25000 [===>..........................] - ETA: 52s - loss: 7.5935 - accuracy: 0.5048
 3808/25000 [===>..........................] - ETA: 52s - loss: 7.5901 - accuracy: 0.5050
 3840/25000 [===>..........................] - ETA: 51s - loss: 7.5828 - accuracy: 0.5055
 3872/25000 [===>..........................] - ETA: 51s - loss: 7.5914 - accuracy: 0.5049
 3904/25000 [===>..........................] - ETA: 51s - loss: 7.5802 - accuracy: 0.5056
 3936/25000 [===>..........................] - ETA: 51s - loss: 7.5887 - accuracy: 0.5051
 3968/25000 [===>..........................] - ETA: 51s - loss: 7.6009 - accuracy: 0.5043
 4000/25000 [===>..........................] - ETA: 51s - loss: 7.6015 - accuracy: 0.5042
 4032/25000 [===>..........................] - ETA: 51s - loss: 7.6020 - accuracy: 0.5042
 4064/25000 [===>..........................] - ETA: 51s - loss: 7.5912 - accuracy: 0.5049
 4096/25000 [===>..........................] - ETA: 51s - loss: 7.5918 - accuracy: 0.5049
 4128/25000 [===>..........................] - ETA: 51s - loss: 7.5812 - accuracy: 0.5056
 4160/25000 [===>..........................] - ETA: 51s - loss: 7.5855 - accuracy: 0.5053
 4192/25000 [====>.........................] - ETA: 50s - loss: 7.6081 - accuracy: 0.5038
 4224/25000 [====>.........................] - ETA: 50s - loss: 7.6085 - accuracy: 0.5038
 4256/25000 [====>.........................] - ETA: 50s - loss: 7.6018 - accuracy: 0.5042
 4288/25000 [====>.........................] - ETA: 50s - loss: 7.5844 - accuracy: 0.5054
 4320/25000 [====>.........................] - ETA: 50s - loss: 7.5956 - accuracy: 0.5046
 4352/25000 [====>.........................] - ETA: 50s - loss: 7.5891 - accuracy: 0.5051
 4384/25000 [====>.........................] - ETA: 50s - loss: 7.5967 - accuracy: 0.5046
 4416/25000 [====>.........................] - ETA: 50s - loss: 7.6006 - accuracy: 0.5043
 4448/25000 [====>.........................] - ETA: 50s - loss: 7.5735 - accuracy: 0.5061
 4480/25000 [====>.........................] - ETA: 50s - loss: 7.5776 - accuracy: 0.5058
 4512/25000 [====>.........................] - ETA: 50s - loss: 7.5715 - accuracy: 0.5062
 4544/25000 [====>.........................] - ETA: 50s - loss: 7.5688 - accuracy: 0.5064
 4576/25000 [====>.........................] - ETA: 49s - loss: 7.5694 - accuracy: 0.5063
 4608/25000 [====>.........................] - ETA: 49s - loss: 7.5635 - accuracy: 0.5067
 4640/25000 [====>.........................] - ETA: 49s - loss: 7.5477 - accuracy: 0.5078
 4672/25000 [====>.........................] - ETA: 49s - loss: 7.5550 - accuracy: 0.5073
 4704/25000 [====>.........................] - ETA: 49s - loss: 7.5525 - accuracy: 0.5074
 4736/25000 [====>.........................] - ETA: 49s - loss: 7.5371 - accuracy: 0.5084
 4768/25000 [====>.........................] - ETA: 49s - loss: 7.5573 - accuracy: 0.5071
 4800/25000 [====>.........................] - ETA: 49s - loss: 7.5325 - accuracy: 0.5088
 4832/25000 [====>.........................] - ETA: 49s - loss: 7.5365 - accuracy: 0.5085
 4864/25000 [====>.........................] - ETA: 49s - loss: 7.5437 - accuracy: 0.5080
 4896/25000 [====>.........................] - ETA: 48s - loss: 7.5664 - accuracy: 0.5065
 4928/25000 [====>.........................] - ETA: 48s - loss: 7.5733 - accuracy: 0.5061
 4960/25000 [====>.........................] - ETA: 48s - loss: 7.5739 - accuracy: 0.5060
 4992/25000 [====>.........................] - ETA: 48s - loss: 7.5683 - accuracy: 0.5064
 5024/25000 [=====>........................] - ETA: 48s - loss: 7.5659 - accuracy: 0.5066
 5056/25000 [=====>........................] - ETA: 48s - loss: 7.5453 - accuracy: 0.5079
 5088/25000 [=====>........................] - ETA: 48s - loss: 7.5310 - accuracy: 0.5088
 5120/25000 [=====>........................] - ETA: 48s - loss: 7.5378 - accuracy: 0.5084
 5152/25000 [=====>........................] - ETA: 48s - loss: 7.5208 - accuracy: 0.5095
 5184/25000 [=====>........................] - ETA: 48s - loss: 7.5306 - accuracy: 0.5089
 5216/25000 [=====>........................] - ETA: 48s - loss: 7.5255 - accuracy: 0.5092
 5248/25000 [=====>........................] - ETA: 47s - loss: 7.5293 - accuracy: 0.5090
 5280/25000 [=====>........................] - ETA: 47s - loss: 7.5127 - accuracy: 0.5100
 5312/25000 [=====>........................] - ETA: 47s - loss: 7.5021 - accuracy: 0.5107
 5344/25000 [=====>........................] - ETA: 47s - loss: 7.5002 - accuracy: 0.5109
 5376/25000 [=====>........................] - ETA: 47s - loss: 7.4983 - accuracy: 0.5110
 5408/25000 [=====>........................] - ETA: 47s - loss: 7.5050 - accuracy: 0.5105
 5440/25000 [=====>........................] - ETA: 47s - loss: 7.5031 - accuracy: 0.5107
 5472/25000 [=====>........................] - ETA: 47s - loss: 7.4985 - accuracy: 0.5110
 5504/25000 [=====>........................] - ETA: 47s - loss: 7.4883 - accuracy: 0.5116
 5536/25000 [=====>........................] - ETA: 47s - loss: 7.4894 - accuracy: 0.5116
 5568/25000 [=====>........................] - ETA: 47s - loss: 7.4959 - accuracy: 0.5111
 5600/25000 [=====>........................] - ETA: 46s - loss: 7.5078 - accuracy: 0.5104
 5632/25000 [=====>........................] - ETA: 46s - loss: 7.5142 - accuracy: 0.5099
 5664/25000 [=====>........................] - ETA: 46s - loss: 7.4988 - accuracy: 0.5109
 5696/25000 [=====>........................] - ETA: 46s - loss: 7.4997 - accuracy: 0.5109
 5728/25000 [=====>........................] - ETA: 46s - loss: 7.5087 - accuracy: 0.5103
 5760/25000 [=====>........................] - ETA: 46s - loss: 7.5122 - accuracy: 0.5101
 5792/25000 [=====>........................] - ETA: 46s - loss: 7.5131 - accuracy: 0.5100
 5824/25000 [=====>........................] - ETA: 46s - loss: 7.5008 - accuracy: 0.5108
 5856/25000 [======>.......................] - ETA: 46s - loss: 7.5095 - accuracy: 0.5102
 5888/25000 [======>.......................] - ETA: 46s - loss: 7.5208 - accuracy: 0.5095
 5920/25000 [======>.......................] - ETA: 46s - loss: 7.5164 - accuracy: 0.5098
 5952/25000 [======>.......................] - ETA: 46s - loss: 7.5146 - accuracy: 0.5099
 5984/25000 [======>.......................] - ETA: 45s - loss: 7.5180 - accuracy: 0.5097
 6016/25000 [======>.......................] - ETA: 45s - loss: 7.5162 - accuracy: 0.5098
 6048/25000 [======>.......................] - ETA: 45s - loss: 7.5221 - accuracy: 0.5094
 6080/25000 [======>.......................] - ETA: 45s - loss: 7.5153 - accuracy: 0.5099
 6112/25000 [======>.......................] - ETA: 45s - loss: 7.5111 - accuracy: 0.5101
 6144/25000 [======>.......................] - ETA: 45s - loss: 7.5119 - accuracy: 0.5101
 6176/25000 [======>.......................] - ETA: 45s - loss: 7.5127 - accuracy: 0.5100
 6208/25000 [======>.......................] - ETA: 45s - loss: 7.5011 - accuracy: 0.5108
 6240/25000 [======>.......................] - ETA: 45s - loss: 7.5094 - accuracy: 0.5103
 6272/25000 [======>.......................] - ETA: 45s - loss: 7.5126 - accuracy: 0.5100
 6304/25000 [======>.......................] - ETA: 45s - loss: 7.5182 - accuracy: 0.5097
 6336/25000 [======>.......................] - ETA: 44s - loss: 7.5238 - accuracy: 0.5093
 6368/25000 [======>.......................] - ETA: 44s - loss: 7.5221 - accuracy: 0.5094
 6400/25000 [======>.......................] - ETA: 44s - loss: 7.5325 - accuracy: 0.5088
 6432/25000 [======>.......................] - ETA: 44s - loss: 7.5379 - accuracy: 0.5084
 6464/25000 [======>.......................] - ETA: 44s - loss: 7.5338 - accuracy: 0.5087
 6496/25000 [======>.......................] - ETA: 44s - loss: 7.5439 - accuracy: 0.5080
 6528/25000 [======>.......................] - ETA: 44s - loss: 7.5515 - accuracy: 0.5075
 6560/25000 [======>.......................] - ETA: 44s - loss: 7.5591 - accuracy: 0.5070
 6592/25000 [======>.......................] - ETA: 44s - loss: 7.5550 - accuracy: 0.5073
 6624/25000 [======>.......................] - ETA: 44s - loss: 7.5578 - accuracy: 0.5071
 6656/25000 [======>.......................] - ETA: 44s - loss: 7.5514 - accuracy: 0.5075
 6688/25000 [=======>......................] - ETA: 44s - loss: 7.5566 - accuracy: 0.5072
 6720/25000 [=======>......................] - ETA: 43s - loss: 7.5434 - accuracy: 0.5080
 6752/25000 [=======>......................] - ETA: 43s - loss: 7.5508 - accuracy: 0.5076
 6784/25000 [=======>......................] - ETA: 43s - loss: 7.5604 - accuracy: 0.5069
 6816/25000 [=======>......................] - ETA: 43s - loss: 7.5654 - accuracy: 0.5066
 6848/25000 [=======>......................] - ETA: 43s - loss: 7.5681 - accuracy: 0.5064
 6880/25000 [=======>......................] - ETA: 43s - loss: 7.5775 - accuracy: 0.5058
 6912/25000 [=======>......................] - ETA: 43s - loss: 7.5801 - accuracy: 0.5056
 6944/25000 [=======>......................] - ETA: 43s - loss: 7.5695 - accuracy: 0.5063
 6976/25000 [=======>......................] - ETA: 43s - loss: 7.5677 - accuracy: 0.5065
 7008/25000 [=======>......................] - ETA: 43s - loss: 7.5616 - accuracy: 0.5068
 7040/25000 [=======>......................] - ETA: 43s - loss: 7.5664 - accuracy: 0.5065
 7072/25000 [=======>......................] - ETA: 43s - loss: 7.5647 - accuracy: 0.5066
 7104/25000 [=======>......................] - ETA: 42s - loss: 7.5587 - accuracy: 0.5070
 7136/25000 [=======>......................] - ETA: 42s - loss: 7.5656 - accuracy: 0.5066
 7168/25000 [=======>......................] - ETA: 42s - loss: 7.5597 - accuracy: 0.5070
 7200/25000 [=======>......................] - ETA: 42s - loss: 7.5644 - accuracy: 0.5067
 7232/25000 [=======>......................] - ETA: 42s - loss: 7.5670 - accuracy: 0.5065
 7264/25000 [=======>......................] - ETA: 42s - loss: 7.5653 - accuracy: 0.5066
 7296/25000 [=======>......................] - ETA: 42s - loss: 7.5636 - accuracy: 0.5067
 7328/25000 [=======>......................] - ETA: 42s - loss: 7.5787 - accuracy: 0.5057
 7360/25000 [=======>......................] - ETA: 42s - loss: 7.5791 - accuracy: 0.5057
 7392/25000 [=======>......................] - ETA: 42s - loss: 7.5816 - accuracy: 0.5055
 7424/25000 [=======>......................] - ETA: 42s - loss: 7.5778 - accuracy: 0.5058
 7456/25000 [=======>......................] - ETA: 42s - loss: 7.5782 - accuracy: 0.5058
 7488/25000 [=======>......................] - ETA: 41s - loss: 7.5765 - accuracy: 0.5059
 7520/25000 [========>.....................] - ETA: 41s - loss: 7.5728 - accuracy: 0.5061
 7552/25000 [========>.....................] - ETA: 41s - loss: 7.5712 - accuracy: 0.5062
 7584/25000 [========>.....................] - ETA: 41s - loss: 7.5716 - accuracy: 0.5062
 7616/25000 [========>.....................] - ETA: 41s - loss: 7.5700 - accuracy: 0.5063
 7648/25000 [========>.....................] - ETA: 41s - loss: 7.5724 - accuracy: 0.5061
 7680/25000 [========>.....................] - ETA: 41s - loss: 7.5708 - accuracy: 0.5063
 7712/25000 [========>.....................] - ETA: 41s - loss: 7.5791 - accuracy: 0.5057
 7744/25000 [========>.....................] - ETA: 41s - loss: 7.5835 - accuracy: 0.5054
 7776/25000 [========>.....................] - ETA: 41s - loss: 7.5838 - accuracy: 0.5054
 7808/25000 [========>.....................] - ETA: 41s - loss: 7.5802 - accuracy: 0.5056
 7840/25000 [========>.....................] - ETA: 41s - loss: 7.5845 - accuracy: 0.5054
 7872/25000 [========>.....................] - ETA: 40s - loss: 7.5926 - accuracy: 0.5048
 7904/25000 [========>.....................] - ETA: 40s - loss: 7.5968 - accuracy: 0.5046
 7936/25000 [========>.....................] - ETA: 40s - loss: 7.6048 - accuracy: 0.5040
 7968/25000 [========>.....................] - ETA: 40s - loss: 7.5973 - accuracy: 0.5045
 8000/25000 [========>.....................] - ETA: 40s - loss: 7.5976 - accuracy: 0.5045
 8032/25000 [========>.....................] - ETA: 40s - loss: 7.5979 - accuracy: 0.5045
 8064/25000 [========>.....................] - ETA: 40s - loss: 7.6039 - accuracy: 0.5041
 8096/25000 [========>.....................] - ETA: 40s - loss: 7.6060 - accuracy: 0.5040
 8128/25000 [========>.....................] - ETA: 40s - loss: 7.6138 - accuracy: 0.5034
 8160/25000 [========>.....................] - ETA: 40s - loss: 7.6159 - accuracy: 0.5033
 8192/25000 [========>.....................] - ETA: 40s - loss: 7.6292 - accuracy: 0.5024
 8224/25000 [========>.....................] - ETA: 40s - loss: 7.6349 - accuracy: 0.5021
 8256/25000 [========>.....................] - ETA: 40s - loss: 7.6332 - accuracy: 0.5022
 8288/25000 [========>.....................] - ETA: 39s - loss: 7.6315 - accuracy: 0.5023
 8320/25000 [========>.....................] - ETA: 39s - loss: 7.6316 - accuracy: 0.5023
 8352/25000 [=========>....................] - ETA: 39s - loss: 7.6317 - accuracy: 0.5023
 8384/25000 [=========>....................] - ETA: 39s - loss: 7.6246 - accuracy: 0.5027
 8416/25000 [=========>....................] - ETA: 39s - loss: 7.6302 - accuracy: 0.5024
 8448/25000 [=========>....................] - ETA: 39s - loss: 7.6376 - accuracy: 0.5019
 8480/25000 [=========>....................] - ETA: 39s - loss: 7.6413 - accuracy: 0.5017
 8512/25000 [=========>....................] - ETA: 39s - loss: 7.6414 - accuracy: 0.5016
 8544/25000 [=========>....................] - ETA: 39s - loss: 7.6361 - accuracy: 0.5020
 8576/25000 [=========>....................] - ETA: 39s - loss: 7.6309 - accuracy: 0.5023
 8608/25000 [=========>....................] - ETA: 39s - loss: 7.6310 - accuracy: 0.5023
 8640/25000 [=========>....................] - ETA: 39s - loss: 7.6418 - accuracy: 0.5016
 8672/25000 [=========>....................] - ETA: 38s - loss: 7.6419 - accuracy: 0.5016
 8704/25000 [=========>....................] - ETA: 38s - loss: 7.6402 - accuracy: 0.5017
 8736/25000 [=========>....................] - ETA: 38s - loss: 7.6420 - accuracy: 0.5016
 8768/25000 [=========>....................] - ETA: 38s - loss: 7.6491 - accuracy: 0.5011
 8800/25000 [=========>....................] - ETA: 38s - loss: 7.6527 - accuracy: 0.5009
 8832/25000 [=========>....................] - ETA: 38s - loss: 7.6475 - accuracy: 0.5012
 8864/25000 [=========>....................] - ETA: 38s - loss: 7.6424 - accuracy: 0.5016
 8896/25000 [=========>....................] - ETA: 38s - loss: 7.6494 - accuracy: 0.5011
 8928/25000 [=========>....................] - ETA: 38s - loss: 7.6477 - accuracy: 0.5012
 8960/25000 [=========>....................] - ETA: 38s - loss: 7.6427 - accuracy: 0.5016
 8992/25000 [=========>....................] - ETA: 38s - loss: 7.6410 - accuracy: 0.5017
 9024/25000 [=========>....................] - ETA: 38s - loss: 7.6428 - accuracy: 0.5016
 9056/25000 [=========>....................] - ETA: 38s - loss: 7.6412 - accuracy: 0.5017
 9088/25000 [=========>....................] - ETA: 37s - loss: 7.6329 - accuracy: 0.5022
 9120/25000 [=========>....................] - ETA: 37s - loss: 7.6347 - accuracy: 0.5021
 9152/25000 [=========>....................] - ETA: 37s - loss: 7.6381 - accuracy: 0.5019
 9184/25000 [==========>...................] - ETA: 37s - loss: 7.6349 - accuracy: 0.5021
 9216/25000 [==========>...................] - ETA: 37s - loss: 7.6417 - accuracy: 0.5016
 9248/25000 [==========>...................] - ETA: 37s - loss: 7.6351 - accuracy: 0.5021
 9280/25000 [==========>...................] - ETA: 37s - loss: 7.6369 - accuracy: 0.5019
 9312/25000 [==========>...................] - ETA: 37s - loss: 7.6386 - accuracy: 0.5018
 9344/25000 [==========>...................] - ETA: 37s - loss: 7.6404 - accuracy: 0.5017
 9376/25000 [==========>...................] - ETA: 37s - loss: 7.6405 - accuracy: 0.5017
 9408/25000 [==========>...................] - ETA: 37s - loss: 7.6422 - accuracy: 0.5016
 9440/25000 [==========>...................] - ETA: 37s - loss: 7.6439 - accuracy: 0.5015
 9472/25000 [==========>...................] - ETA: 37s - loss: 7.6391 - accuracy: 0.5018
 9504/25000 [==========>...................] - ETA: 36s - loss: 7.6424 - accuracy: 0.5016
 9536/25000 [==========>...................] - ETA: 36s - loss: 7.6457 - accuracy: 0.5014
 9568/25000 [==========>...................] - ETA: 36s - loss: 7.6458 - accuracy: 0.5014
 9600/25000 [==========>...................] - ETA: 36s - loss: 7.6459 - accuracy: 0.5014
 9632/25000 [==========>...................] - ETA: 36s - loss: 7.6443 - accuracy: 0.5015
 9664/25000 [==========>...................] - ETA: 36s - loss: 7.6412 - accuracy: 0.5017
 9696/25000 [==========>...................] - ETA: 36s - loss: 7.6429 - accuracy: 0.5015
 9728/25000 [==========>...................] - ETA: 36s - loss: 7.6493 - accuracy: 0.5011
 9760/25000 [==========>...................] - ETA: 36s - loss: 7.6493 - accuracy: 0.5011
 9792/25000 [==========>...................] - ETA: 36s - loss: 7.6447 - accuracy: 0.5014
 9824/25000 [==========>...................] - ETA: 36s - loss: 7.6432 - accuracy: 0.5015
 9856/25000 [==========>...................] - ETA: 36s - loss: 7.6417 - accuracy: 0.5016
 9888/25000 [==========>...................] - ETA: 35s - loss: 7.6434 - accuracy: 0.5015
 9920/25000 [==========>...................] - ETA: 35s - loss: 7.6403 - accuracy: 0.5017
 9952/25000 [==========>...................] - ETA: 35s - loss: 7.6481 - accuracy: 0.5012
 9984/25000 [==========>...................] - ETA: 35s - loss: 7.6436 - accuracy: 0.5015
10016/25000 [===========>..................] - ETA: 35s - loss: 7.6452 - accuracy: 0.5014
10048/25000 [===========>..................] - ETA: 35s - loss: 7.6422 - accuracy: 0.5016
10080/25000 [===========>..................] - ETA: 35s - loss: 7.6438 - accuracy: 0.5015
10112/25000 [===========>..................] - ETA: 35s - loss: 7.6454 - accuracy: 0.5014
10144/25000 [===========>..................] - ETA: 35s - loss: 7.6470 - accuracy: 0.5013
10176/25000 [===========>..................] - ETA: 35s - loss: 7.6485 - accuracy: 0.5012
10208/25000 [===========>..................] - ETA: 35s - loss: 7.6531 - accuracy: 0.5009
10240/25000 [===========>..................] - ETA: 35s - loss: 7.6576 - accuracy: 0.5006
10272/25000 [===========>..................] - ETA: 35s - loss: 7.6606 - accuracy: 0.5004
10304/25000 [===========>..................] - ETA: 34s - loss: 7.6622 - accuracy: 0.5003
10336/25000 [===========>..................] - ETA: 34s - loss: 7.6607 - accuracy: 0.5004
10368/25000 [===========>..................] - ETA: 34s - loss: 7.6622 - accuracy: 0.5003
10400/25000 [===========>..................] - ETA: 34s - loss: 7.6622 - accuracy: 0.5003
10432/25000 [===========>..................] - ETA: 34s - loss: 7.6740 - accuracy: 0.4995
10464/25000 [===========>..................] - ETA: 34s - loss: 7.6754 - accuracy: 0.4994
10496/25000 [===========>..................] - ETA: 34s - loss: 7.6754 - accuracy: 0.4994
10528/25000 [===========>..................] - ETA: 34s - loss: 7.6797 - accuracy: 0.4991
10560/25000 [===========>..................] - ETA: 34s - loss: 7.6869 - accuracy: 0.4987
10592/25000 [===========>..................] - ETA: 34s - loss: 7.6869 - accuracy: 0.4987
10624/25000 [===========>..................] - ETA: 34s - loss: 7.6897 - accuracy: 0.4985
10656/25000 [===========>..................] - ETA: 34s - loss: 7.6911 - accuracy: 0.4984
10688/25000 [===========>..................] - ETA: 33s - loss: 7.6910 - accuracy: 0.4984
10720/25000 [===========>..................] - ETA: 33s - loss: 7.6924 - accuracy: 0.4983
10752/25000 [===========>..................] - ETA: 33s - loss: 7.6951 - accuracy: 0.4981
10784/25000 [===========>..................] - ETA: 33s - loss: 7.6993 - accuracy: 0.4979
10816/25000 [===========>..................] - ETA: 33s - loss: 7.7021 - accuracy: 0.4977
10848/25000 [============>.................] - ETA: 33s - loss: 7.7005 - accuracy: 0.4978
10880/25000 [============>.................] - ETA: 33s - loss: 7.6990 - accuracy: 0.4979
10912/25000 [============>.................] - ETA: 33s - loss: 7.6849 - accuracy: 0.4988
10944/25000 [============>.................] - ETA: 33s - loss: 7.6778 - accuracy: 0.4993
10976/25000 [============>.................] - ETA: 33s - loss: 7.6750 - accuracy: 0.4995
11008/25000 [============>.................] - ETA: 33s - loss: 7.6708 - accuracy: 0.4997
11040/25000 [============>.................] - ETA: 33s - loss: 7.6722 - accuracy: 0.4996
11072/25000 [============>.................] - ETA: 33s - loss: 7.6805 - accuracy: 0.4991
11104/25000 [============>.................] - ETA: 32s - loss: 7.6721 - accuracy: 0.4996
11136/25000 [============>.................] - ETA: 32s - loss: 7.6776 - accuracy: 0.4993
11168/25000 [============>.................] - ETA: 32s - loss: 7.6803 - accuracy: 0.4991
11200/25000 [============>.................] - ETA: 32s - loss: 7.6776 - accuracy: 0.4993
11232/25000 [============>.................] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
11264/25000 [============>.................] - ETA: 32s - loss: 7.6761 - accuracy: 0.4994
11296/25000 [============>.................] - ETA: 32s - loss: 7.6734 - accuracy: 0.4996
11328/25000 [============>.................] - ETA: 32s - loss: 7.6774 - accuracy: 0.4993
11360/25000 [============>.................] - ETA: 32s - loss: 7.6720 - accuracy: 0.4996
11392/25000 [============>.................] - ETA: 32s - loss: 7.6693 - accuracy: 0.4998
11424/25000 [============>.................] - ETA: 32s - loss: 7.6706 - accuracy: 0.4997
11456/25000 [============>.................] - ETA: 32s - loss: 7.6693 - accuracy: 0.4998
11488/25000 [============>.................] - ETA: 32s - loss: 7.6653 - accuracy: 0.5001
11520/25000 [============>.................] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
11552/25000 [============>.................] - ETA: 31s - loss: 7.6613 - accuracy: 0.5003
11584/25000 [============>.................] - ETA: 31s - loss: 7.6613 - accuracy: 0.5003
11616/25000 [============>.................] - ETA: 31s - loss: 7.6640 - accuracy: 0.5002
11648/25000 [============>.................] - ETA: 31s - loss: 7.6614 - accuracy: 0.5003
11680/25000 [=============>................] - ETA: 31s - loss: 7.6679 - accuracy: 0.4999
11712/25000 [=============>................] - ETA: 31s - loss: 7.6653 - accuracy: 0.5001
11744/25000 [=============>................] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
11776/25000 [=============>................] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
11808/25000 [=============>................] - ETA: 31s - loss: 7.6718 - accuracy: 0.4997
11840/25000 [=============>................] - ETA: 31s - loss: 7.6679 - accuracy: 0.4999
11872/25000 [=============>................] - ETA: 31s - loss: 7.6653 - accuracy: 0.5001
11904/25000 [=============>................] - ETA: 31s - loss: 7.6640 - accuracy: 0.5002
11936/25000 [=============>................] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
11968/25000 [=============>................] - ETA: 30s - loss: 7.6705 - accuracy: 0.4997
12000/25000 [=============>................] - ETA: 30s - loss: 7.6768 - accuracy: 0.4993
12032/25000 [=============>................] - ETA: 30s - loss: 7.6794 - accuracy: 0.4992
12064/25000 [=============>................] - ETA: 30s - loss: 7.6857 - accuracy: 0.4988
12096/25000 [=============>................] - ETA: 30s - loss: 7.6894 - accuracy: 0.4985
12128/25000 [=============>................] - ETA: 30s - loss: 7.6843 - accuracy: 0.4988
12160/25000 [=============>................] - ETA: 30s - loss: 7.6805 - accuracy: 0.4991
12192/25000 [=============>................] - ETA: 30s - loss: 7.6754 - accuracy: 0.4994
12224/25000 [=============>................] - ETA: 30s - loss: 7.6741 - accuracy: 0.4995
12256/25000 [=============>................] - ETA: 30s - loss: 7.6704 - accuracy: 0.4998
12288/25000 [=============>................] - ETA: 30s - loss: 7.6704 - accuracy: 0.4998
12320/25000 [=============>................] - ETA: 30s - loss: 7.6691 - accuracy: 0.4998
12352/25000 [=============>................] - ETA: 29s - loss: 7.6703 - accuracy: 0.4998
12384/25000 [=============>................] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
12416/25000 [=============>................] - ETA: 29s - loss: 7.6691 - accuracy: 0.4998
12448/25000 [=============>................] - ETA: 29s - loss: 7.6654 - accuracy: 0.5001
12480/25000 [=============>................] - ETA: 29s - loss: 7.6654 - accuracy: 0.5001
12512/25000 [==============>...............] - ETA: 29s - loss: 7.6691 - accuracy: 0.4998
12544/25000 [==============>...............] - ETA: 29s - loss: 7.6727 - accuracy: 0.4996
12576/25000 [==============>...............] - ETA: 29s - loss: 7.6752 - accuracy: 0.4994
12608/25000 [==============>...............] - ETA: 29s - loss: 7.6788 - accuracy: 0.4992
12640/25000 [==============>...............] - ETA: 29s - loss: 7.6751 - accuracy: 0.4994
12672/25000 [==============>...............] - ETA: 29s - loss: 7.6763 - accuracy: 0.4994
12704/25000 [==============>...............] - ETA: 29s - loss: 7.6787 - accuracy: 0.4992
12736/25000 [==============>...............] - ETA: 29s - loss: 7.6750 - accuracy: 0.4995
12768/25000 [==============>...............] - ETA: 28s - loss: 7.6774 - accuracy: 0.4993
12800/25000 [==============>...............] - ETA: 28s - loss: 7.6750 - accuracy: 0.4995
12832/25000 [==============>...............] - ETA: 28s - loss: 7.6750 - accuracy: 0.4995
12864/25000 [==============>...............] - ETA: 28s - loss: 7.6785 - accuracy: 0.4992
12896/25000 [==============>...............] - ETA: 28s - loss: 7.6749 - accuracy: 0.4995
12928/25000 [==============>...............] - ETA: 28s - loss: 7.6773 - accuracy: 0.4993
12960/25000 [==============>...............] - ETA: 28s - loss: 7.6785 - accuracy: 0.4992
12992/25000 [==============>...............] - ETA: 28s - loss: 7.6772 - accuracy: 0.4993
13024/25000 [==============>...............] - ETA: 28s - loss: 7.6807 - accuracy: 0.4991
13056/25000 [==============>...............] - ETA: 28s - loss: 7.6819 - accuracy: 0.4990
13088/25000 [==============>...............] - ETA: 28s - loss: 7.6889 - accuracy: 0.4985
13120/25000 [==============>...............] - ETA: 28s - loss: 7.6853 - accuracy: 0.4988
13152/25000 [==============>...............] - ETA: 28s - loss: 7.6888 - accuracy: 0.4986
13184/25000 [==============>...............] - ETA: 27s - loss: 7.6910 - accuracy: 0.4984
13216/25000 [==============>...............] - ETA: 27s - loss: 7.6910 - accuracy: 0.4984
13248/25000 [==============>...............] - ETA: 27s - loss: 7.6875 - accuracy: 0.4986
13280/25000 [==============>...............] - ETA: 27s - loss: 7.6828 - accuracy: 0.4989
13312/25000 [==============>...............] - ETA: 27s - loss: 7.6781 - accuracy: 0.4992
13344/25000 [===============>..............] - ETA: 27s - loss: 7.6770 - accuracy: 0.4993
13376/25000 [===============>..............] - ETA: 27s - loss: 7.6735 - accuracy: 0.4996
13408/25000 [===============>..............] - ETA: 27s - loss: 7.6723 - accuracy: 0.4996
13440/25000 [===============>..............] - ETA: 27s - loss: 7.6746 - accuracy: 0.4995
13472/25000 [===============>..............] - ETA: 27s - loss: 7.6712 - accuracy: 0.4997
13504/25000 [===============>..............] - ETA: 27s - loss: 7.6723 - accuracy: 0.4996
13536/25000 [===============>..............] - ETA: 27s - loss: 7.6689 - accuracy: 0.4999
13568/25000 [===============>..............] - ETA: 26s - loss: 7.6711 - accuracy: 0.4997
13600/25000 [===============>..............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
13632/25000 [===============>..............] - ETA: 26s - loss: 7.6655 - accuracy: 0.5001
13664/25000 [===============>..............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
13696/25000 [===============>..............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
13728/25000 [===============>..............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
13760/25000 [===============>..............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
13792/25000 [===============>..............] - ETA: 26s - loss: 7.6722 - accuracy: 0.4996
13824/25000 [===============>..............] - ETA: 26s - loss: 7.6733 - accuracy: 0.4996
13856/25000 [===============>..............] - ETA: 26s - loss: 7.6722 - accuracy: 0.4996
13888/25000 [===============>..............] - ETA: 26s - loss: 7.6699 - accuracy: 0.4998
13920/25000 [===============>..............] - ETA: 26s - loss: 7.6710 - accuracy: 0.4997
13952/25000 [===============>..............] - ETA: 26s - loss: 7.6787 - accuracy: 0.4992
13984/25000 [===============>..............] - ETA: 25s - loss: 7.6699 - accuracy: 0.4998
14016/25000 [===============>..............] - ETA: 25s - loss: 7.6688 - accuracy: 0.4999
14048/25000 [===============>..............] - ETA: 25s - loss: 7.6721 - accuracy: 0.4996
14080/25000 [===============>..............] - ETA: 25s - loss: 7.6732 - accuracy: 0.4996
14112/25000 [===============>..............] - ETA: 25s - loss: 7.6742 - accuracy: 0.4995
14144/25000 [===============>..............] - ETA: 25s - loss: 7.6742 - accuracy: 0.4995
14176/25000 [================>.............] - ETA: 25s - loss: 7.6742 - accuracy: 0.4995
14208/25000 [================>.............] - ETA: 25s - loss: 7.6763 - accuracy: 0.4994
14240/25000 [================>.............] - ETA: 25s - loss: 7.6806 - accuracy: 0.4991
14272/25000 [================>.............] - ETA: 25s - loss: 7.6817 - accuracy: 0.4990
14304/25000 [================>.............] - ETA: 25s - loss: 7.6752 - accuracy: 0.4994
14336/25000 [================>.............] - ETA: 25s - loss: 7.6773 - accuracy: 0.4993
14368/25000 [================>.............] - ETA: 25s - loss: 7.6794 - accuracy: 0.4992
14400/25000 [================>.............] - ETA: 24s - loss: 7.6741 - accuracy: 0.4995
14432/25000 [================>.............] - ETA: 24s - loss: 7.6719 - accuracy: 0.4997
14464/25000 [================>.............] - ETA: 24s - loss: 7.6730 - accuracy: 0.4996
14496/25000 [================>.............] - ETA: 24s - loss: 7.6740 - accuracy: 0.4995
14528/25000 [================>.............] - ETA: 24s - loss: 7.6719 - accuracy: 0.4997
14560/25000 [================>.............] - ETA: 24s - loss: 7.6677 - accuracy: 0.4999
14592/25000 [================>.............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
14624/25000 [================>.............] - ETA: 24s - loss: 7.6677 - accuracy: 0.4999
14656/25000 [================>.............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
14688/25000 [================>.............] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
14720/25000 [================>.............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
14752/25000 [================>.............] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
14784/25000 [================>.............] - ETA: 24s - loss: 7.6645 - accuracy: 0.5001
14816/25000 [================>.............] - ETA: 24s - loss: 7.6594 - accuracy: 0.5005
14848/25000 [================>.............] - ETA: 23s - loss: 7.6635 - accuracy: 0.5002
14880/25000 [================>.............] - ETA: 23s - loss: 7.6687 - accuracy: 0.4999
14912/25000 [================>.............] - ETA: 23s - loss: 7.6718 - accuracy: 0.4997
14944/25000 [================>.............] - ETA: 23s - loss: 7.6728 - accuracy: 0.4996
14976/25000 [================>.............] - ETA: 23s - loss: 7.6789 - accuracy: 0.4992
15008/25000 [=================>............] - ETA: 23s - loss: 7.6768 - accuracy: 0.4993
15040/25000 [=================>............] - ETA: 23s - loss: 7.6727 - accuracy: 0.4996
15072/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15104/25000 [=================>............] - ETA: 23s - loss: 7.6686 - accuracy: 0.4999
15136/25000 [=================>............] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
15168/25000 [=================>............] - ETA: 23s - loss: 7.6686 - accuracy: 0.4999
15200/25000 [=================>............] - ETA: 23s - loss: 7.6636 - accuracy: 0.5002
15232/25000 [=================>............] - ETA: 23s - loss: 7.6576 - accuracy: 0.5006
15264/25000 [=================>............] - ETA: 22s - loss: 7.6556 - accuracy: 0.5007
15296/25000 [=================>............] - ETA: 22s - loss: 7.6556 - accuracy: 0.5007
15328/25000 [=================>............] - ETA: 22s - loss: 7.6516 - accuracy: 0.5010
15360/25000 [=================>............] - ETA: 22s - loss: 7.6536 - accuracy: 0.5008
15392/25000 [=================>............] - ETA: 22s - loss: 7.6596 - accuracy: 0.5005
15424/25000 [=================>............] - ETA: 22s - loss: 7.6607 - accuracy: 0.5004
15456/25000 [=================>............] - ETA: 22s - loss: 7.6627 - accuracy: 0.5003
15488/25000 [=================>............] - ETA: 22s - loss: 7.6607 - accuracy: 0.5004
15520/25000 [=================>............] - ETA: 22s - loss: 7.6577 - accuracy: 0.5006
15552/25000 [=================>............] - ETA: 22s - loss: 7.6548 - accuracy: 0.5008
15584/25000 [=================>............] - ETA: 22s - loss: 7.6509 - accuracy: 0.5010
15616/25000 [=================>............] - ETA: 22s - loss: 7.6470 - accuracy: 0.5013
15648/25000 [=================>............] - ETA: 22s - loss: 7.6421 - accuracy: 0.5016
15680/25000 [=================>............] - ETA: 21s - loss: 7.6432 - accuracy: 0.5015
15712/25000 [=================>............] - ETA: 21s - loss: 7.6451 - accuracy: 0.5014
15744/25000 [=================>............] - ETA: 21s - loss: 7.6471 - accuracy: 0.5013
15776/25000 [=================>............] - ETA: 21s - loss: 7.6452 - accuracy: 0.5014
15808/25000 [=================>............] - ETA: 21s - loss: 7.6385 - accuracy: 0.5018
15840/25000 [==================>...........] - ETA: 21s - loss: 7.6405 - accuracy: 0.5017
15872/25000 [==================>...........] - ETA: 21s - loss: 7.6376 - accuracy: 0.5019
15904/25000 [==================>...........] - ETA: 21s - loss: 7.6425 - accuracy: 0.5016
15936/25000 [==================>...........] - ETA: 21s - loss: 7.6455 - accuracy: 0.5014
15968/25000 [==================>...........] - ETA: 21s - loss: 7.6445 - accuracy: 0.5014
16000/25000 [==================>...........] - ETA: 21s - loss: 7.6446 - accuracy: 0.5014
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6446 - accuracy: 0.5014
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6437 - accuracy: 0.5015
16096/25000 [==================>...........] - ETA: 20s - loss: 7.6447 - accuracy: 0.5014
16128/25000 [==================>...........] - ETA: 20s - loss: 7.6457 - accuracy: 0.5014
16160/25000 [==================>...........] - ETA: 20s - loss: 7.6448 - accuracy: 0.5014
16192/25000 [==================>...........] - ETA: 20s - loss: 7.6401 - accuracy: 0.5017
16224/25000 [==================>...........] - ETA: 20s - loss: 7.6326 - accuracy: 0.5022
16256/25000 [==================>...........] - ETA: 20s - loss: 7.6327 - accuracy: 0.5022
16288/25000 [==================>...........] - ETA: 20s - loss: 7.6403 - accuracy: 0.5017
16320/25000 [==================>...........] - ETA: 20s - loss: 7.6441 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 20s - loss: 7.6451 - accuracy: 0.5014
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6432 - accuracy: 0.5015
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6386 - accuracy: 0.5018
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6377 - accuracy: 0.5019
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6415 - accuracy: 0.5016
16512/25000 [==================>...........] - ETA: 19s - loss: 7.6406 - accuracy: 0.5017
16544/25000 [==================>...........] - ETA: 19s - loss: 7.6397 - accuracy: 0.5018
16576/25000 [==================>...........] - ETA: 19s - loss: 7.6370 - accuracy: 0.5019
16608/25000 [==================>...........] - ETA: 19s - loss: 7.6389 - accuracy: 0.5018
16640/25000 [==================>...........] - ETA: 19s - loss: 7.6362 - accuracy: 0.5020
16672/25000 [===================>..........] - ETA: 19s - loss: 7.6344 - accuracy: 0.5021
16704/25000 [===================>..........] - ETA: 19s - loss: 7.6345 - accuracy: 0.5021
16736/25000 [===================>..........] - ETA: 19s - loss: 7.6364 - accuracy: 0.5020
16768/25000 [===================>..........] - ETA: 19s - loss: 7.6355 - accuracy: 0.5020
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6365 - accuracy: 0.5020
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6393 - accuracy: 0.5018
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6403 - accuracy: 0.5017
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6448 - accuracy: 0.5014
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6422 - accuracy: 0.5016
16960/25000 [===================>..........] - ETA: 18s - loss: 7.6404 - accuracy: 0.5017
16992/25000 [===================>..........] - ETA: 18s - loss: 7.6368 - accuracy: 0.5019
17024/25000 [===================>..........] - ETA: 18s - loss: 7.6324 - accuracy: 0.5022
17056/25000 [===================>..........] - ETA: 18s - loss: 7.6298 - accuracy: 0.5024
17088/25000 [===================>..........] - ETA: 18s - loss: 7.6271 - accuracy: 0.5026
17120/25000 [===================>..........] - ETA: 18s - loss: 7.6272 - accuracy: 0.5026
17152/25000 [===================>..........] - ETA: 18s - loss: 7.6273 - accuracy: 0.5026
17184/25000 [===================>..........] - ETA: 18s - loss: 7.6202 - accuracy: 0.5030
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6212 - accuracy: 0.5030
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6195 - accuracy: 0.5031
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6196 - accuracy: 0.5031
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6179 - accuracy: 0.5032
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6145 - accuracy: 0.5034
17376/25000 [===================>..........] - ETA: 17s - loss: 7.6119 - accuracy: 0.5036
17408/25000 [===================>..........] - ETA: 17s - loss: 7.6111 - accuracy: 0.5036
17440/25000 [===================>..........] - ETA: 17s - loss: 7.6121 - accuracy: 0.5036
17472/25000 [===================>..........] - ETA: 17s - loss: 7.6148 - accuracy: 0.5034
17504/25000 [====================>.........] - ETA: 17s - loss: 7.6114 - accuracy: 0.5036
17536/25000 [====================>.........] - ETA: 17s - loss: 7.6124 - accuracy: 0.5035
17568/25000 [====================>.........] - ETA: 17s - loss: 7.6143 - accuracy: 0.5034
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6152 - accuracy: 0.5034
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6162 - accuracy: 0.5033
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6171 - accuracy: 0.5032
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6207 - accuracy: 0.5030
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6199 - accuracy: 0.5030
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6217 - accuracy: 0.5029
17792/25000 [====================>.........] - ETA: 16s - loss: 7.6184 - accuracy: 0.5031
17824/25000 [====================>.........] - ETA: 16s - loss: 7.6167 - accuracy: 0.5033
17856/25000 [====================>.........] - ETA: 16s - loss: 7.6177 - accuracy: 0.5032
17888/25000 [====================>.........] - ETA: 16s - loss: 7.6238 - accuracy: 0.5028
17920/25000 [====================>.........] - ETA: 16s - loss: 7.6196 - accuracy: 0.5031
17952/25000 [====================>.........] - ETA: 16s - loss: 7.6214 - accuracy: 0.5030
17984/25000 [====================>.........] - ETA: 16s - loss: 7.6240 - accuracy: 0.5028
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6232 - accuracy: 0.5028
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6267 - accuracy: 0.5026
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6285 - accuracy: 0.5025
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6260 - accuracy: 0.5027
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6269 - accuracy: 0.5026
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6312 - accuracy: 0.5023
18208/25000 [====================>.........] - ETA: 15s - loss: 7.6321 - accuracy: 0.5023
18240/25000 [====================>.........] - ETA: 15s - loss: 7.6330 - accuracy: 0.5022
18272/25000 [====================>.........] - ETA: 15s - loss: 7.6339 - accuracy: 0.5021
18304/25000 [====================>.........] - ETA: 15s - loss: 7.6323 - accuracy: 0.5022
18336/25000 [=====================>........] - ETA: 15s - loss: 7.6315 - accuracy: 0.5023
18368/25000 [=====================>........] - ETA: 15s - loss: 7.6307 - accuracy: 0.5023
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6266 - accuracy: 0.5026
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6292 - accuracy: 0.5024
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6301 - accuracy: 0.5024
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6285 - accuracy: 0.5025
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6302 - accuracy: 0.5024
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6319 - accuracy: 0.5023
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6320 - accuracy: 0.5023
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6329 - accuracy: 0.5022
18656/25000 [=====================>........] - ETA: 14s - loss: 7.6288 - accuracy: 0.5025
18688/25000 [=====================>........] - ETA: 14s - loss: 7.6281 - accuracy: 0.5025
18720/25000 [=====================>........] - ETA: 14s - loss: 7.6281 - accuracy: 0.5025
18752/25000 [=====================>........] - ETA: 14s - loss: 7.6274 - accuracy: 0.5026
18784/25000 [=====================>........] - ETA: 14s - loss: 7.6258 - accuracy: 0.5027
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6259 - accuracy: 0.5027
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6251 - accuracy: 0.5027
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6260 - accuracy: 0.5026
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6269 - accuracy: 0.5026
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6270 - accuracy: 0.5026
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6294 - accuracy: 0.5024
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6319 - accuracy: 0.5023
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6320 - accuracy: 0.5023
19072/25000 [=====================>........] - ETA: 13s - loss: 7.6312 - accuracy: 0.5023
19104/25000 [=====================>........] - ETA: 13s - loss: 7.6305 - accuracy: 0.5024
19136/25000 [=====================>........] - ETA: 13s - loss: 7.6314 - accuracy: 0.5023
19168/25000 [======================>.......] - ETA: 13s - loss: 7.6322 - accuracy: 0.5022
19200/25000 [======================>.......] - ETA: 13s - loss: 7.6307 - accuracy: 0.5023
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6291 - accuracy: 0.5024
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6284 - accuracy: 0.5025
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6293 - accuracy: 0.5024
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6277 - accuracy: 0.5025
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6310 - accuracy: 0.5023
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6279 - accuracy: 0.5025
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6264 - accuracy: 0.5026
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6280 - accuracy: 0.5025
19488/25000 [======================>.......] - ETA: 12s - loss: 7.6289 - accuracy: 0.5025
19520/25000 [======================>.......] - ETA: 12s - loss: 7.6313 - accuracy: 0.5023
19552/25000 [======================>.......] - ETA: 12s - loss: 7.6290 - accuracy: 0.5025
19584/25000 [======================>.......] - ETA: 12s - loss: 7.6314 - accuracy: 0.5023
19616/25000 [======================>.......] - ETA: 12s - loss: 7.6291 - accuracy: 0.5024
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6284 - accuracy: 0.5025
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6269 - accuracy: 0.5026
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6269 - accuracy: 0.5026
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6278 - accuracy: 0.5025
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6294 - accuracy: 0.5024
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6271 - accuracy: 0.5026
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6241 - accuracy: 0.5028
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6265 - accuracy: 0.5026
19904/25000 [======================>.......] - ETA: 11s - loss: 7.6273 - accuracy: 0.5026
19936/25000 [======================>.......] - ETA: 11s - loss: 7.6282 - accuracy: 0.5025
19968/25000 [======================>.......] - ETA: 11s - loss: 7.6267 - accuracy: 0.5026
20000/25000 [=======================>......] - ETA: 11s - loss: 7.6252 - accuracy: 0.5027
20032/25000 [=======================>......] - ETA: 11s - loss: 7.6291 - accuracy: 0.5024
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6307 - accuracy: 0.5023
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6323 - accuracy: 0.5022
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6323 - accuracy: 0.5022
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6339 - accuracy: 0.5021
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6355 - accuracy: 0.5020
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6371 - accuracy: 0.5019
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6379 - accuracy: 0.5019
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6409 - accuracy: 0.5017
20320/25000 [=======================>......] - ETA: 10s - loss: 7.6410 - accuracy: 0.5017
20352/25000 [=======================>......] - ETA: 10s - loss: 7.6410 - accuracy: 0.5017
20384/25000 [=======================>......] - ETA: 10s - loss: 7.6425 - accuracy: 0.5016
20416/25000 [=======================>......] - ETA: 10s - loss: 7.6448 - accuracy: 0.5014
20448/25000 [=======================>......] - ETA: 10s - loss: 7.6426 - accuracy: 0.5016
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6457 - accuracy: 0.5014
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6427 - accuracy: 0.5016
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6420 - accuracy: 0.5016
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6405 - accuracy: 0.5017
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6398 - accuracy: 0.5017
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6391 - accuracy: 0.5018
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6369 - accuracy: 0.5019
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6385 - accuracy: 0.5018
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6393 - accuracy: 0.5018
20768/25000 [=======================>......] - ETA: 9s - loss: 7.6400 - accuracy: 0.5017 
20800/25000 [=======================>......] - ETA: 9s - loss: 7.6393 - accuracy: 0.5018
20832/25000 [=======================>......] - ETA: 9s - loss: 7.6423 - accuracy: 0.5016
20864/25000 [========================>.....] - ETA: 9s - loss: 7.6402 - accuracy: 0.5017
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6417 - accuracy: 0.5016
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6417 - accuracy: 0.5016
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6432 - accuracy: 0.5015
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6454 - accuracy: 0.5014
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6477 - accuracy: 0.5012
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6470 - accuracy: 0.5013
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6434 - accuracy: 0.5015
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6419 - accuracy: 0.5016
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6463 - accuracy: 0.5013
21184/25000 [========================>.....] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
21216/25000 [========================>.....] - ETA: 8s - loss: 7.6420 - accuracy: 0.5016
21248/25000 [========================>.....] - ETA: 8s - loss: 7.6428 - accuracy: 0.5016
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6428 - accuracy: 0.5016
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6472 - accuracy: 0.5013
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6508 - accuracy: 0.5010
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6508 - accuracy: 0.5010
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6516 - accuracy: 0.5010
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6523 - accuracy: 0.5009
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6523 - accuracy: 0.5009
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6524 - accuracy: 0.5009
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6524 - accuracy: 0.5009
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6531 - accuracy: 0.5009
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6496 - accuracy: 0.5011
21632/25000 [========================>.....] - ETA: 7s - loss: 7.6503 - accuracy: 0.5011
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6503 - accuracy: 0.5011
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6482 - accuracy: 0.5012
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6469 - accuracy: 0.5013
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6490 - accuracy: 0.5011
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6483 - accuracy: 0.5012
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6491 - accuracy: 0.5011
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6484 - accuracy: 0.5012
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6498 - accuracy: 0.5011
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6512 - accuracy: 0.5010
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6555 - accuracy: 0.5007
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6576 - accuracy: 0.5006
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6569 - accuracy: 0.5006
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6569 - accuracy: 0.5006
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6611 - accuracy: 0.5004
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6583 - accuracy: 0.5005
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6583 - accuracy: 0.5005
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6597 - accuracy: 0.5005
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6639 - accuracy: 0.5002
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6687 - accuracy: 0.4999
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6687 - accuracy: 0.4999
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6680 - accuracy: 0.4999
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6680 - accuracy: 0.4999
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6721 - accuracy: 0.4996
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6720 - accuracy: 0.4996
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6693 - accuracy: 0.4998
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6680 - accuracy: 0.4999
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6713 - accuracy: 0.4997
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6706 - accuracy: 0.4997
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6640 - accuracy: 0.5002
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 1s - loss: 7.6749 - accuracy: 0.4995
24224/25000 [============================>.] - ETA: 1s - loss: 7.6767 - accuracy: 0.4993
24256/25000 [============================>.] - ETA: 1s - loss: 7.6767 - accuracy: 0.4993
24288/25000 [============================>.] - ETA: 1s - loss: 7.6742 - accuracy: 0.4995
24320/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24352/25000 [============================>.] - ETA: 1s - loss: 7.6761 - accuracy: 0.4994
24384/25000 [============================>.] - ETA: 1s - loss: 7.6805 - accuracy: 0.4991
24416/25000 [============================>.] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
24448/25000 [============================>.] - ETA: 1s - loss: 7.6798 - accuracy: 0.4991
24480/25000 [============================>.] - ETA: 1s - loss: 7.6773 - accuracy: 0.4993
24512/25000 [============================>.] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
24544/25000 [============================>.] - ETA: 1s - loss: 7.6766 - accuracy: 0.4993
24576/25000 [============================>.] - ETA: 0s - loss: 7.6754 - accuracy: 0.4994
24608/25000 [============================>.] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
24640/25000 [============================>.] - ETA: 0s - loss: 7.6735 - accuracy: 0.4996
24672/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24704/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24768/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6709 - accuracy: 0.4997
24832/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 69s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
Loading data...





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb to script
[NbConvertApp] Writing 1800 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/sklearn_titanic_randomForest_example2.py
Deprecaton set to False
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.py", line 22, in <module>
    pars = json.load(open( data_path , mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//mnist_mlmodels_.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb to script
[NbConvertApp] Writing 7241 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/mnist_mlmodels_.txt
python: can't open file '/home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.py': [Errno 2] No such file or directory
No replacement





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl_titanic.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb to script
[NbConvertApp] Writing 1434 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/gluon_automl_titanic.py
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.py", line 27, in <module>
    data_path= '../mlmodels/dataset/json/gluon_automl.json'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py", line 82, in get_params
    with open(data_path, encoding='utf-8') as config_f:
FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow__lstm_json.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb to script
[NbConvertApp] Writing 1379 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/tensorflow__lstm_json.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.py", line 13, in <module>
    print( os.getcwd())
NameError: name 'os' is not defined





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb to script
[NbConvertApp] Writing 1070 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/sklearn.py
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.py", line 34, in <module>
    module        =  module_load( model_uri= model_uri )                           # Load file definition
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_titanic.ipynb 

[NbConvertApp] Converting notebook /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb to script
[NbConvertApp] Writing 1355 bytes to /home/runner/work/mlmodels/mlmodels/mlmodels/example/lightgbm_titanic.py
Deprecaton set to False
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.py", line 21, in <module>
    pars = json.load(open( data_path , mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.py 

  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py", line 15
    !git clone https://github.com/ahmed3bbas/mlmodels.git
    ^
SyntaxError: invalid syntax





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m4.py 






 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_hyper.py 

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py", line 5, in <module>
    print(mlmodels)
NameError: name 'mlmodels' is not defined





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.py 

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py", line 16, in <module>
    print( os.getcwd())
NameError: name 'os' is not defined





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m5.py 

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py", line 27, in <module>
    import mxnet as mx
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
    from . import contrib
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
    from . import onnx
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
    from .onnx2mx.import_model import import_model, get_model_metadata
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
    from . import import_model
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
    from .import_onnx import GraphProto
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
    from ._import_helper import _convert_map as convert_map
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
    from . import _translation_utils as translation_utils
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
    from .... import  module
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
    from .base_module import BaseModule
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
    from ..model import BatchEndParam
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
    from sklearn.base import BaseEstimator
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example/sklearn.py", line 34, in <module>
    module        =  module_load( model_uri= model_uri )                           # Load file definition
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_model.py 

<module 'mlmodels' from '/home/runner/work/mlmodels/mlmodels/mlmodels/__init__.py'>
/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/ardmn.json
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py", line 27, in <module>
    pars = json.load(open(config_path , mode='r'))[config_mode]
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/ardmn.json'





 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m4.py 






 ************************************************************************************************************************
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m5.py 

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 72, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py", line 27, in <module>
    import mxnet as mx
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
    from . import contrib
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
    from . import onnx
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
    from .onnx2mx.import_model import import_model, get_model_metadata
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
    from . import import_model
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
    from .import_onnx import GraphProto
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
    from ._import_helper import _convert_map as convert_map
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
    from . import _translation_utils as translation_utils
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
    from .... import  module
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
    from .base_module import BaseModule
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
    from ..model import BatchEndParam
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
    from sklearn.base import BaseEstimator
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/example/sklearn.py", line 34, in <module>
    module        =  module_load( model_uri= model_uri )                           # Load file definition
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range

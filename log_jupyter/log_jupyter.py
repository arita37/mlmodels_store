
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:50<01:15, 25.11s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.14491672411370551, 'embedding_size_factor': 1.2972300140000375, 'layers.choice': 2, 'learning_rate': 0.0005398065256412438, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 3.694623215968357e-07} and reward: 0.381
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc2\x8c\xa1\x97[\x1c\xe9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xc1tBXQ\x08X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?A\xb09\xaf\xd8\x90\x8bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x98\xcbPyw\xbd\x9bu.' and reward: 0.381
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc2\x8c\xa1\x97[\x1c\xe9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xc1tBXQ\x08X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?A\xb09\xaf\xd8\x90\x8bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x98\xcbPyw\xbd\x9bu.' and reward: 0.381
 60%|██████    | 3/5 [01:40<01:05, 32.53s/it] 60%|██████    | 3/5 [01:40<01:06, 33.36s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.462421831982292, 'embedding_size_factor': 1.1887443984236796, 'layers.choice': 2, 'learning_rate': 0.0009083177465008804, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.5927389894862583e-12} and reward: 0.3654
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\x98Q\xbdT\x80\xa1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x05\x18\xd8\xa8\x86\xc0X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?M\xc3\x85\x81\xff\x1a\xfdX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=|\x05\x0f\x08>\x8d\x9eu.' and reward: 0.3654
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\x98Q\xbdT\x80\xa1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x05\x18\xd8\xa8\x86\xc0X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?M\xc3\x85\x81\xff\x1a\xfdX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=|\x05\x0f\x08>\x8d\x9eu.' and reward: 0.3654
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 151.87623476982117
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -34.53s of remaining time.
Ensemble size: 38
Ensemble weights: 
[0.44736842 0.21052632 0.34210526]
	0.3918	 = Validation accuracy score
	1.04s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 155.61s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
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
 2252800/17464789 [==>...........................] - ETA: 0s
 9609216/17464789 [===============>..............] - ETA: 0s
16211968/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-25 10:20:34.946777: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-25 10:20:34.951348: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-25 10:20:34.951482: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ea4fc956a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 10:20:34.951496: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:31 - loss: 8.6249 - accuracy: 0.4375
   64/25000 [..............................] - ETA: 2:45 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:08 - loss: 7.9861 - accuracy: 0.4792
  128/25000 [..............................] - ETA: 1:52 - loss: 8.5052 - accuracy: 0.4453
  160/25000 [..............................] - ETA: 1:41 - loss: 8.6249 - accuracy: 0.4375
  192/25000 [..............................] - ETA: 1:34 - loss: 8.8645 - accuracy: 0.4219
  224/25000 [..............................] - ETA: 1:29 - loss: 8.7619 - accuracy: 0.4286
  256/25000 [..............................] - ETA: 1:25 - loss: 8.5052 - accuracy: 0.4453
  288/25000 [..............................] - ETA: 1:22 - loss: 8.4120 - accuracy: 0.4514
  320/25000 [..............................] - ETA: 1:20 - loss: 8.3374 - accuracy: 0.4563
  352/25000 [..............................] - ETA: 1:18 - loss: 8.1458 - accuracy: 0.4688
  384/25000 [..............................] - ETA: 1:17 - loss: 8.0260 - accuracy: 0.4766
  416/25000 [..............................] - ETA: 1:15 - loss: 7.9983 - accuracy: 0.4784
  448/25000 [..............................] - ETA: 1:14 - loss: 7.9747 - accuracy: 0.4799
  480/25000 [..............................] - ETA: 1:13 - loss: 7.9541 - accuracy: 0.4812
  512/25000 [..............................] - ETA: 1:12 - loss: 7.9960 - accuracy: 0.4785
  544/25000 [..............................] - ETA: 1:11 - loss: 8.0894 - accuracy: 0.4724
  576/25000 [..............................] - ETA: 1:11 - loss: 7.9328 - accuracy: 0.4826
  608/25000 [..............................] - ETA: 1:10 - loss: 7.8684 - accuracy: 0.4868
  640/25000 [..............................] - ETA: 1:09 - loss: 7.9062 - accuracy: 0.4844
  672/25000 [..............................] - ETA: 1:08 - loss: 7.8035 - accuracy: 0.4911
  704/25000 [..............................] - ETA: 1:08 - loss: 7.8409 - accuracy: 0.4886
  736/25000 [..............................] - ETA: 1:07 - loss: 7.9375 - accuracy: 0.4823
  768/25000 [..............................] - ETA: 1:07 - loss: 7.9861 - accuracy: 0.4792
  800/25000 [..............................] - ETA: 1:07 - loss: 7.9733 - accuracy: 0.4800
  832/25000 [..............................] - ETA: 1:07 - loss: 8.0536 - accuracy: 0.4748
  864/25000 [>.............................] - ETA: 1:06 - loss: 8.0216 - accuracy: 0.4769
  896/25000 [>.............................] - ETA: 1:06 - loss: 8.0089 - accuracy: 0.4777
  928/25000 [>.............................] - ETA: 1:05 - loss: 8.0466 - accuracy: 0.4752
  960/25000 [>.............................] - ETA: 1:05 - loss: 8.0979 - accuracy: 0.4719
  992/25000 [>.............................] - ETA: 1:05 - loss: 8.0994 - accuracy: 0.4718
 1024/25000 [>.............................] - ETA: 1:04 - loss: 8.0709 - accuracy: 0.4736
 1056/25000 [>.............................] - ETA: 1:04 - loss: 8.0732 - accuracy: 0.4735
 1088/25000 [>.............................] - ETA: 1:04 - loss: 8.0330 - accuracy: 0.4761
 1120/25000 [>.............................] - ETA: 1:04 - loss: 8.0363 - accuracy: 0.4759
 1152/25000 [>.............................] - ETA: 1:03 - loss: 7.9728 - accuracy: 0.4800
 1184/25000 [>.............................] - ETA: 1:03 - loss: 8.0551 - accuracy: 0.4747
 1216/25000 [>.............................] - ETA: 1:03 - loss: 8.0701 - accuracy: 0.4737
 1248/25000 [>.............................] - ETA: 1:02 - loss: 8.0352 - accuracy: 0.4760
 1280/25000 [>.............................] - ETA: 1:02 - loss: 8.0020 - accuracy: 0.4781
 1312/25000 [>.............................] - ETA: 1:02 - loss: 8.0757 - accuracy: 0.4733
 1344/25000 [>.............................] - ETA: 1:02 - loss: 8.0317 - accuracy: 0.4762
 1376/25000 [>.............................] - ETA: 1:02 - loss: 8.0343 - accuracy: 0.4760
 1408/25000 [>.............................] - ETA: 1:02 - loss: 8.0042 - accuracy: 0.4780
 1440/25000 [>.............................] - ETA: 1:01 - loss: 7.9648 - accuracy: 0.4806
 1472/25000 [>.............................] - ETA: 1:01 - loss: 7.9687 - accuracy: 0.4803
 1504/25000 [>.............................] - ETA: 1:01 - loss: 7.9827 - accuracy: 0.4794
 1536/25000 [>.............................] - ETA: 1:01 - loss: 8.0060 - accuracy: 0.4779
 1568/25000 [>.............................] - ETA: 1:01 - loss: 7.9991 - accuracy: 0.4783
 1600/25000 [>.............................] - ETA: 1:00 - loss: 7.9733 - accuracy: 0.4800
 1632/25000 [>.............................] - ETA: 1:00 - loss: 8.0236 - accuracy: 0.4767
 1664/25000 [>.............................] - ETA: 1:00 - loss: 8.0536 - accuracy: 0.4748
 1696/25000 [=>............................] - ETA: 1:00 - loss: 8.0554 - accuracy: 0.4746
 1728/25000 [=>............................] - ETA: 1:00 - loss: 8.0659 - accuracy: 0.4740
 1760/25000 [=>............................] - ETA: 1:00 - loss: 8.0325 - accuracy: 0.4761
 1792/25000 [=>............................] - ETA: 1:00 - loss: 8.0602 - accuracy: 0.4743
 1824/25000 [=>............................] - ETA: 59s - loss: 8.0701 - accuracy: 0.4737 
 1856/25000 [=>............................] - ETA: 59s - loss: 8.0219 - accuracy: 0.4768
 1888/25000 [=>............................] - ETA: 59s - loss: 8.0240 - accuracy: 0.4767
 1920/25000 [=>............................] - ETA: 59s - loss: 8.0100 - accuracy: 0.4776
 1952/25000 [=>............................] - ETA: 59s - loss: 8.0201 - accuracy: 0.4769
 1984/25000 [=>............................] - ETA: 59s - loss: 8.0221 - accuracy: 0.4768
 2016/25000 [=>............................] - ETA: 59s - loss: 8.0089 - accuracy: 0.4777
 2048/25000 [=>............................] - ETA: 59s - loss: 8.0110 - accuracy: 0.4775
 2080/25000 [=>............................] - ETA: 58s - loss: 8.0278 - accuracy: 0.4764
 2112/25000 [=>............................] - ETA: 58s - loss: 8.0224 - accuracy: 0.4768
 2144/25000 [=>............................] - ETA: 58s - loss: 8.0171 - accuracy: 0.4771
 2176/25000 [=>............................] - ETA: 58s - loss: 8.0189 - accuracy: 0.4770
 2208/25000 [=>............................] - ETA: 58s - loss: 8.0138 - accuracy: 0.4774
 2240/25000 [=>............................] - ETA: 58s - loss: 7.9952 - accuracy: 0.4786
 2272/25000 [=>............................] - ETA: 58s - loss: 7.9973 - accuracy: 0.4784
 2304/25000 [=>............................] - ETA: 58s - loss: 8.0127 - accuracy: 0.4774
 2336/25000 [=>............................] - ETA: 58s - loss: 8.0342 - accuracy: 0.4760
 2368/25000 [=>............................] - ETA: 57s - loss: 8.0487 - accuracy: 0.4751
 2400/25000 [=>............................] - ETA: 57s - loss: 8.0180 - accuracy: 0.4771
 2432/25000 [=>............................] - ETA: 57s - loss: 8.0134 - accuracy: 0.4774
 2464/25000 [=>............................] - ETA: 57s - loss: 8.0338 - accuracy: 0.4761
 2496/25000 [=>............................] - ETA: 57s - loss: 8.0229 - accuracy: 0.4768
 2528/25000 [==>...........................] - ETA: 57s - loss: 8.0548 - accuracy: 0.4747
 2560/25000 [==>...........................] - ETA: 57s - loss: 8.0559 - accuracy: 0.4746
 2592/25000 [==>...........................] - ETA: 56s - loss: 8.0570 - accuracy: 0.4745
 2624/25000 [==>...........................] - ETA: 56s - loss: 8.0464 - accuracy: 0.4752
 2656/25000 [==>...........................] - ETA: 56s - loss: 8.0534 - accuracy: 0.4748
 2688/25000 [==>...........................] - ETA: 56s - loss: 8.0716 - accuracy: 0.4736
 2720/25000 [==>...........................] - ETA: 56s - loss: 8.0725 - accuracy: 0.4735
 2752/25000 [==>...........................] - ETA: 56s - loss: 8.0734 - accuracy: 0.4735
 2784/25000 [==>...........................] - ETA: 56s - loss: 8.0466 - accuracy: 0.4752
 2816/25000 [==>...........................] - ETA: 56s - loss: 8.0369 - accuracy: 0.4759
 2848/25000 [==>...........................] - ETA: 56s - loss: 8.0381 - accuracy: 0.4758
 2880/25000 [==>...........................] - ETA: 55s - loss: 8.0180 - accuracy: 0.4771
 2912/25000 [==>...........................] - ETA: 55s - loss: 7.9931 - accuracy: 0.4787
 2944/25000 [==>...........................] - ETA: 55s - loss: 7.9843 - accuracy: 0.4793
 2976/25000 [==>...........................] - ETA: 55s - loss: 7.9964 - accuracy: 0.4785
 3008/25000 [==>...........................] - ETA: 55s - loss: 7.9929 - accuracy: 0.4787
 3040/25000 [==>...........................] - ETA: 55s - loss: 7.9995 - accuracy: 0.4783
 3072/25000 [==>...........................] - ETA: 55s - loss: 8.0160 - accuracy: 0.4772
 3104/25000 [==>...........................] - ETA: 55s - loss: 8.0124 - accuracy: 0.4774
 3136/25000 [==>...........................] - ETA: 55s - loss: 7.9698 - accuracy: 0.4802
 3168/25000 [==>...........................] - ETA: 54s - loss: 7.9812 - accuracy: 0.4795
 3200/25000 [==>...........................] - ETA: 54s - loss: 7.9829 - accuracy: 0.4794
 3232/25000 [==>...........................] - ETA: 54s - loss: 7.9940 - accuracy: 0.4787
 3264/25000 [==>...........................] - ETA: 54s - loss: 8.0049 - accuracy: 0.4779
 3296/25000 [==>...........................] - ETA: 54s - loss: 8.0248 - accuracy: 0.4766
 3328/25000 [==>...........................] - ETA: 54s - loss: 8.0122 - accuracy: 0.4775
 3360/25000 [===>..........................] - ETA: 54s - loss: 8.0043 - accuracy: 0.4780
 3392/25000 [===>..........................] - ETA: 54s - loss: 8.0011 - accuracy: 0.4782
 3424/25000 [===>..........................] - ETA: 54s - loss: 8.0249 - accuracy: 0.4766
 3456/25000 [===>..........................] - ETA: 53s - loss: 7.9949 - accuracy: 0.4786
 3488/25000 [===>..........................] - ETA: 53s - loss: 8.0007 - accuracy: 0.4782
 3520/25000 [===>..........................] - ETA: 53s - loss: 8.0020 - accuracy: 0.4781
 3552/25000 [===>..........................] - ETA: 53s - loss: 8.0033 - accuracy: 0.4780
 3584/25000 [===>..........................] - ETA: 53s - loss: 7.9875 - accuracy: 0.4791
 3616/25000 [===>..........................] - ETA: 53s - loss: 7.9931 - accuracy: 0.4787
 3648/25000 [===>..........................] - ETA: 53s - loss: 7.9945 - accuracy: 0.4786
 3680/25000 [===>..........................] - ETA: 53s - loss: 7.9958 - accuracy: 0.4785
 3712/25000 [===>..........................] - ETA: 53s - loss: 7.9764 - accuracy: 0.4798
 3744/25000 [===>..........................] - ETA: 52s - loss: 7.9779 - accuracy: 0.4797
 3776/25000 [===>..........................] - ETA: 52s - loss: 7.9631 - accuracy: 0.4807
 3808/25000 [===>..........................] - ETA: 52s - loss: 7.9686 - accuracy: 0.4803
 3840/25000 [===>..........................] - ETA: 52s - loss: 7.9661 - accuracy: 0.4805
 3872/25000 [===>..........................] - ETA: 52s - loss: 7.9755 - accuracy: 0.4799
 3904/25000 [===>..........................] - ETA: 52s - loss: 7.9651 - accuracy: 0.4805
 3936/25000 [===>..........................] - ETA: 52s - loss: 7.9354 - accuracy: 0.4825
 3968/25000 [===>..........................] - ETA: 52s - loss: 7.9217 - accuracy: 0.4834
 4000/25000 [===>..........................] - ETA: 52s - loss: 7.9311 - accuracy: 0.4827
 4032/25000 [===>..........................] - ETA: 52s - loss: 7.9290 - accuracy: 0.4829
 4064/25000 [===>..........................] - ETA: 51s - loss: 7.9194 - accuracy: 0.4835
 4096/25000 [===>..........................] - ETA: 51s - loss: 7.9287 - accuracy: 0.4829
 4128/25000 [===>..........................] - ETA: 51s - loss: 7.9415 - accuracy: 0.4821
 4160/25000 [===>..........................] - ETA: 51s - loss: 7.9283 - accuracy: 0.4829
 4192/25000 [====>.........................] - ETA: 51s - loss: 7.9409 - accuracy: 0.4821
 4224/25000 [====>.........................] - ETA: 51s - loss: 7.9425 - accuracy: 0.4820
 4256/25000 [====>.........................] - ETA: 51s - loss: 7.9332 - accuracy: 0.4826
 4288/25000 [====>.........................] - ETA: 51s - loss: 7.9277 - accuracy: 0.4830
 4320/25000 [====>.........................] - ETA: 51s - loss: 7.9151 - accuracy: 0.4838
 4352/25000 [====>.........................] - ETA: 51s - loss: 7.8992 - accuracy: 0.4848
 4384/25000 [====>.........................] - ETA: 50s - loss: 7.8835 - accuracy: 0.4859
 4416/25000 [====>.........................] - ETA: 50s - loss: 7.8819 - accuracy: 0.4860
 4448/25000 [====>.........................] - ETA: 50s - loss: 7.8735 - accuracy: 0.4865
 4480/25000 [====>.........................] - ETA: 50s - loss: 7.8651 - accuracy: 0.4871
 4512/25000 [====>.........................] - ETA: 50s - loss: 7.8603 - accuracy: 0.4874
 4544/25000 [====>.........................] - ETA: 50s - loss: 7.8488 - accuracy: 0.4881
 4576/25000 [====>.........................] - ETA: 50s - loss: 7.8543 - accuracy: 0.4878
 4608/25000 [====>.........................] - ETA: 50s - loss: 7.8596 - accuracy: 0.4874
 4640/25000 [====>.........................] - ETA: 50s - loss: 7.8550 - accuracy: 0.4877
 4672/25000 [====>.........................] - ETA: 50s - loss: 7.8504 - accuracy: 0.4880
 4704/25000 [====>.........................] - ETA: 50s - loss: 7.8329 - accuracy: 0.4892
 4736/25000 [====>.........................] - ETA: 49s - loss: 7.8253 - accuracy: 0.4897
 4768/25000 [====>.........................] - ETA: 49s - loss: 7.8178 - accuracy: 0.4901
 4800/25000 [====>.........................] - ETA: 49s - loss: 7.8040 - accuracy: 0.4910
 4832/25000 [====>.........................] - ETA: 49s - loss: 7.8031 - accuracy: 0.4911
 4864/25000 [====>.........................] - ETA: 49s - loss: 7.8085 - accuracy: 0.4907
 4896/25000 [====>.........................] - ETA: 49s - loss: 7.7856 - accuracy: 0.4922
 4928/25000 [====>.........................] - ETA: 49s - loss: 7.7911 - accuracy: 0.4919
 4960/25000 [====>.........................] - ETA: 49s - loss: 7.7995 - accuracy: 0.4913
 4992/25000 [====>.........................] - ETA: 49s - loss: 7.8079 - accuracy: 0.4908
 5024/25000 [=====>........................] - ETA: 49s - loss: 7.8131 - accuracy: 0.4904
 5056/25000 [=====>........................] - ETA: 49s - loss: 7.8092 - accuracy: 0.4907
 5088/25000 [=====>........................] - ETA: 49s - loss: 7.8143 - accuracy: 0.4904
 5120/25000 [=====>........................] - ETA: 48s - loss: 7.7924 - accuracy: 0.4918
 5152/25000 [=====>........................] - ETA: 48s - loss: 7.7916 - accuracy: 0.4918
 5184/25000 [=====>........................] - ETA: 48s - loss: 7.7790 - accuracy: 0.4927
 5216/25000 [=====>........................] - ETA: 48s - loss: 7.7901 - accuracy: 0.4919
 5248/25000 [=====>........................] - ETA: 48s - loss: 7.7689 - accuracy: 0.4933
 5280/25000 [=====>........................] - ETA: 48s - loss: 7.7799 - accuracy: 0.4926
 5312/25000 [=====>........................] - ETA: 48s - loss: 7.7734 - accuracy: 0.4930
 5344/25000 [=====>........................] - ETA: 48s - loss: 7.7699 - accuracy: 0.4933
 5376/25000 [=====>........................] - ETA: 48s - loss: 7.7750 - accuracy: 0.4929
 5408/25000 [=====>........................] - ETA: 48s - loss: 7.7630 - accuracy: 0.4937
 5440/25000 [=====>........................] - ETA: 48s - loss: 7.7596 - accuracy: 0.4939
 5472/25000 [=====>........................] - ETA: 47s - loss: 7.7535 - accuracy: 0.4943
 5504/25000 [=====>........................] - ETA: 47s - loss: 7.7558 - accuracy: 0.4942
 5536/25000 [=====>........................] - ETA: 47s - loss: 7.7525 - accuracy: 0.4944
 5568/25000 [=====>........................] - ETA: 47s - loss: 7.7437 - accuracy: 0.4950
 5600/25000 [=====>........................] - ETA: 47s - loss: 7.7542 - accuracy: 0.4943
 5632/25000 [=====>........................] - ETA: 47s - loss: 7.7483 - accuracy: 0.4947
 5664/25000 [=====>........................] - ETA: 47s - loss: 7.7614 - accuracy: 0.4938
 5696/25000 [=====>........................] - ETA: 47s - loss: 7.7528 - accuracy: 0.4944
 5728/25000 [=====>........................] - ETA: 47s - loss: 7.7496 - accuracy: 0.4946
 5760/25000 [=====>........................] - ETA: 47s - loss: 7.7651 - accuracy: 0.4936
 5792/25000 [=====>........................] - ETA: 47s - loss: 7.7699 - accuracy: 0.4933
 5824/25000 [=====>........................] - ETA: 47s - loss: 7.7561 - accuracy: 0.4942
 5856/25000 [======>.......................] - ETA: 46s - loss: 7.7504 - accuracy: 0.4945
 5888/25000 [======>.......................] - ETA: 46s - loss: 7.7604 - accuracy: 0.4939
 5920/25000 [======>.......................] - ETA: 46s - loss: 7.7573 - accuracy: 0.4941
 5952/25000 [======>.......................] - ETA: 46s - loss: 7.7516 - accuracy: 0.4945
 5984/25000 [======>.......................] - ETA: 46s - loss: 7.7461 - accuracy: 0.4948
 6016/25000 [======>.......................] - ETA: 46s - loss: 7.7507 - accuracy: 0.4945
 6048/25000 [======>.......................] - ETA: 46s - loss: 7.7554 - accuracy: 0.4942
 6080/25000 [======>.......................] - ETA: 46s - loss: 7.7423 - accuracy: 0.4951
 6112/25000 [======>.......................] - ETA: 46s - loss: 7.7419 - accuracy: 0.4951
 6144/25000 [======>.......................] - ETA: 46s - loss: 7.7490 - accuracy: 0.4946
 6176/25000 [======>.......................] - ETA: 46s - loss: 7.7461 - accuracy: 0.4948
 6208/25000 [======>.......................] - ETA: 46s - loss: 7.7407 - accuracy: 0.4952
 6240/25000 [======>.......................] - ETA: 45s - loss: 7.7379 - accuracy: 0.4954
 6272/25000 [======>.......................] - ETA: 45s - loss: 7.7400 - accuracy: 0.4952
 6304/25000 [======>.......................] - ETA: 45s - loss: 7.7445 - accuracy: 0.4949
 6336/25000 [======>.......................] - ETA: 45s - loss: 7.7465 - accuracy: 0.4948
 6368/25000 [======>.......................] - ETA: 45s - loss: 7.7605 - accuracy: 0.4939
 6400/25000 [======>.......................] - ETA: 45s - loss: 7.7505 - accuracy: 0.4945
 6432/25000 [======>.......................] - ETA: 45s - loss: 7.7501 - accuracy: 0.4946
 6464/25000 [======>.......................] - ETA: 45s - loss: 7.7544 - accuracy: 0.4943
 6496/25000 [======>.......................] - ETA: 45s - loss: 7.7540 - accuracy: 0.4943
 6528/25000 [======>.......................] - ETA: 45s - loss: 7.7371 - accuracy: 0.4954
 6560/25000 [======>.......................] - ETA: 45s - loss: 7.7344 - accuracy: 0.4956
 6592/25000 [======>.......................] - ETA: 45s - loss: 7.7364 - accuracy: 0.4954
 6624/25000 [======>.......................] - ETA: 44s - loss: 7.7337 - accuracy: 0.4956
 6656/25000 [======>.......................] - ETA: 44s - loss: 7.7334 - accuracy: 0.4956
 6688/25000 [=======>......................] - ETA: 44s - loss: 7.7331 - accuracy: 0.4957
 6720/25000 [=======>......................] - ETA: 44s - loss: 7.7328 - accuracy: 0.4957
 6752/25000 [=======>......................] - ETA: 44s - loss: 7.7370 - accuracy: 0.4954
 6784/25000 [=======>......................] - ETA: 44s - loss: 7.7344 - accuracy: 0.4956
 6816/25000 [=======>......................] - ETA: 44s - loss: 7.7386 - accuracy: 0.4953
 6848/25000 [=======>......................] - ETA: 44s - loss: 7.7383 - accuracy: 0.4953
 6880/25000 [=======>......................] - ETA: 44s - loss: 7.7246 - accuracy: 0.4962
 6912/25000 [=======>......................] - ETA: 44s - loss: 7.7176 - accuracy: 0.4967
 6944/25000 [=======>......................] - ETA: 44s - loss: 7.6997 - accuracy: 0.4978
 6976/25000 [=======>......................] - ETA: 44s - loss: 7.7018 - accuracy: 0.4977
 7008/25000 [=======>......................] - ETA: 44s - loss: 7.7148 - accuracy: 0.4969
 7040/25000 [=======>......................] - ETA: 43s - loss: 7.7102 - accuracy: 0.4972
 7072/25000 [=======>......................] - ETA: 43s - loss: 7.7035 - accuracy: 0.4976
 7104/25000 [=======>......................] - ETA: 43s - loss: 7.7012 - accuracy: 0.4977
 7136/25000 [=======>......................] - ETA: 43s - loss: 7.6967 - accuracy: 0.4980
 7168/25000 [=======>......................] - ETA: 43s - loss: 7.7008 - accuracy: 0.4978
 7200/25000 [=======>......................] - ETA: 43s - loss: 7.6943 - accuracy: 0.4982
 7232/25000 [=======>......................] - ETA: 43s - loss: 7.6921 - accuracy: 0.4983
 7264/25000 [=======>......................] - ETA: 43s - loss: 7.6877 - accuracy: 0.4986
 7296/25000 [=======>......................] - ETA: 43s - loss: 7.6918 - accuracy: 0.4984
 7328/25000 [=======>......................] - ETA: 43s - loss: 7.6729 - accuracy: 0.4996
 7360/25000 [=======>......................] - ETA: 43s - loss: 7.6708 - accuracy: 0.4997
 7392/25000 [=======>......................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7424/25000 [=======>......................] - ETA: 42s - loss: 7.6769 - accuracy: 0.4993
 7456/25000 [=======>......................] - ETA: 42s - loss: 7.6790 - accuracy: 0.4992
 7488/25000 [=======>......................] - ETA: 42s - loss: 7.6748 - accuracy: 0.4995
 7520/25000 [========>.....................] - ETA: 42s - loss: 7.6789 - accuracy: 0.4992
 7552/25000 [========>.....................] - ETA: 42s - loss: 7.6707 - accuracy: 0.4997
 7584/25000 [========>.....................] - ETA: 42s - loss: 7.6747 - accuracy: 0.4995
 7616/25000 [========>.....................] - ETA: 42s - loss: 7.6706 - accuracy: 0.4997
 7648/25000 [========>.....................] - ETA: 42s - loss: 7.6706 - accuracy: 0.4997
 7680/25000 [========>.....................] - ETA: 42s - loss: 7.6686 - accuracy: 0.4999
 7712/25000 [========>.....................] - ETA: 42s - loss: 7.6686 - accuracy: 0.4999
 7744/25000 [========>.....................] - ETA: 42s - loss: 7.6785 - accuracy: 0.4992
 7776/25000 [========>.....................] - ETA: 42s - loss: 7.6706 - accuracy: 0.4997
 7808/25000 [========>.....................] - ETA: 41s - loss: 7.6705 - accuracy: 0.4997
 7840/25000 [========>.....................] - ETA: 41s - loss: 7.6705 - accuracy: 0.4997
 7872/25000 [========>.....................] - ETA: 41s - loss: 7.6647 - accuracy: 0.5001
 7904/25000 [========>.....................] - ETA: 41s - loss: 7.6627 - accuracy: 0.5003
 7936/25000 [========>.....................] - ETA: 41s - loss: 7.6570 - accuracy: 0.5006
 7968/25000 [========>.....................] - ETA: 41s - loss: 7.6608 - accuracy: 0.5004
 8000/25000 [========>.....................] - ETA: 41s - loss: 7.6647 - accuracy: 0.5001
 8032/25000 [========>.....................] - ETA: 41s - loss: 7.6609 - accuracy: 0.5004
 8064/25000 [========>.....................] - ETA: 41s - loss: 7.6571 - accuracy: 0.5006
 8096/25000 [========>.....................] - ETA: 41s - loss: 7.6609 - accuracy: 0.5004
 8128/25000 [========>.....................] - ETA: 41s - loss: 7.6610 - accuracy: 0.5004
 8160/25000 [========>.....................] - ETA: 41s - loss: 7.6591 - accuracy: 0.5005
 8192/25000 [========>.....................] - ETA: 40s - loss: 7.6610 - accuracy: 0.5004
 8224/25000 [========>.....................] - ETA: 40s - loss: 7.6573 - accuracy: 0.5006
 8256/25000 [========>.....................] - ETA: 40s - loss: 7.6629 - accuracy: 0.5002
 8288/25000 [========>.....................] - ETA: 40s - loss: 7.6500 - accuracy: 0.5011
 8320/25000 [========>.....................] - ETA: 40s - loss: 7.6500 - accuracy: 0.5011
 8352/25000 [=========>....................] - ETA: 40s - loss: 7.6538 - accuracy: 0.5008
 8384/25000 [=========>....................] - ETA: 40s - loss: 7.6502 - accuracy: 0.5011
 8416/25000 [=========>....................] - ETA: 40s - loss: 7.6448 - accuracy: 0.5014
 8448/25000 [=========>....................] - ETA: 40s - loss: 7.6503 - accuracy: 0.5011
 8480/25000 [=========>....................] - ETA: 40s - loss: 7.6522 - accuracy: 0.5009
 8512/25000 [=========>....................] - ETA: 40s - loss: 7.6558 - accuracy: 0.5007
 8544/25000 [=========>....................] - ETA: 40s - loss: 7.6505 - accuracy: 0.5011
 8576/25000 [=========>....................] - ETA: 40s - loss: 7.6470 - accuracy: 0.5013
 8608/25000 [=========>....................] - ETA: 39s - loss: 7.6452 - accuracy: 0.5014
 8640/25000 [=========>....................] - ETA: 39s - loss: 7.6418 - accuracy: 0.5016
 8672/25000 [=========>....................] - ETA: 39s - loss: 7.6419 - accuracy: 0.5016
 8704/25000 [=========>....................] - ETA: 39s - loss: 7.6437 - accuracy: 0.5015
 8736/25000 [=========>....................] - ETA: 39s - loss: 7.6491 - accuracy: 0.5011
 8768/25000 [=========>....................] - ETA: 39s - loss: 7.6596 - accuracy: 0.5005
 8800/25000 [=========>....................] - ETA: 39s - loss: 7.6579 - accuracy: 0.5006
 8832/25000 [=========>....................] - ETA: 39s - loss: 7.6579 - accuracy: 0.5006
 8864/25000 [=========>....................] - ETA: 39s - loss: 7.6562 - accuracy: 0.5007
 8896/25000 [=========>....................] - ETA: 39s - loss: 7.6442 - accuracy: 0.5015
 8928/25000 [=========>....................] - ETA: 39s - loss: 7.6460 - accuracy: 0.5013
 8960/25000 [=========>....................] - ETA: 39s - loss: 7.6581 - accuracy: 0.5006
 8992/25000 [=========>....................] - ETA: 38s - loss: 7.6598 - accuracy: 0.5004
 9024/25000 [=========>....................] - ETA: 38s - loss: 7.6615 - accuracy: 0.5003
 9056/25000 [=========>....................] - ETA: 38s - loss: 7.6632 - accuracy: 0.5002
 9088/25000 [=========>....................] - ETA: 38s - loss: 7.6599 - accuracy: 0.5004
 9120/25000 [=========>....................] - ETA: 38s - loss: 7.6616 - accuracy: 0.5003
 9152/25000 [=========>....................] - ETA: 38s - loss: 7.6616 - accuracy: 0.5003
 9184/25000 [==========>...................] - ETA: 38s - loss: 7.6583 - accuracy: 0.5005
 9216/25000 [==========>...................] - ETA: 38s - loss: 7.6516 - accuracy: 0.5010
 9248/25000 [==========>...................] - ETA: 38s - loss: 7.6534 - accuracy: 0.5009
 9280/25000 [==========>...................] - ETA: 38s - loss: 7.6451 - accuracy: 0.5014
 9312/25000 [==========>...................] - ETA: 38s - loss: 7.6403 - accuracy: 0.5017
 9344/25000 [==========>...................] - ETA: 38s - loss: 7.6354 - accuracy: 0.5020
 9376/25000 [==========>...................] - ETA: 37s - loss: 7.6372 - accuracy: 0.5019
 9408/25000 [==========>...................] - ETA: 37s - loss: 7.6389 - accuracy: 0.5018
 9440/25000 [==========>...................] - ETA: 37s - loss: 7.6423 - accuracy: 0.5016
 9472/25000 [==========>...................] - ETA: 37s - loss: 7.6407 - accuracy: 0.5017
 9504/25000 [==========>...................] - ETA: 37s - loss: 7.6360 - accuracy: 0.5020
 9536/25000 [==========>...................] - ETA: 37s - loss: 7.6312 - accuracy: 0.5023
 9568/25000 [==========>...................] - ETA: 37s - loss: 7.6314 - accuracy: 0.5023
 9600/25000 [==========>...................] - ETA: 37s - loss: 7.6283 - accuracy: 0.5025
 9632/25000 [==========>...................] - ETA: 37s - loss: 7.6380 - accuracy: 0.5019
 9664/25000 [==========>...................] - ETA: 37s - loss: 7.6428 - accuracy: 0.5016
 9696/25000 [==========>...................] - ETA: 37s - loss: 7.6413 - accuracy: 0.5017
 9728/25000 [==========>...................] - ETA: 37s - loss: 7.6414 - accuracy: 0.5016
 9760/25000 [==========>...................] - ETA: 37s - loss: 7.6493 - accuracy: 0.5011
 9792/25000 [==========>...................] - ETA: 37s - loss: 7.6557 - accuracy: 0.5007
 9824/25000 [==========>...................] - ETA: 36s - loss: 7.6604 - accuracy: 0.5004
 9856/25000 [==========>...................] - ETA: 36s - loss: 7.6620 - accuracy: 0.5003
 9888/25000 [==========>...................] - ETA: 36s - loss: 7.6651 - accuracy: 0.5001
 9920/25000 [==========>...................] - ETA: 36s - loss: 7.6697 - accuracy: 0.4998
 9952/25000 [==========>...................] - ETA: 36s - loss: 7.6743 - accuracy: 0.4995
 9984/25000 [==========>...................] - ETA: 36s - loss: 7.6728 - accuracy: 0.4996
10016/25000 [===========>..................] - ETA: 36s - loss: 7.6727 - accuracy: 0.4996
10048/25000 [===========>..................] - ETA: 36s - loss: 7.6727 - accuracy: 0.4996
10080/25000 [===========>..................] - ETA: 36s - loss: 7.6742 - accuracy: 0.4995
10112/25000 [===========>..................] - ETA: 36s - loss: 7.6878 - accuracy: 0.4986
10144/25000 [===========>..................] - ETA: 36s - loss: 7.6908 - accuracy: 0.4984
10176/25000 [===========>..................] - ETA: 36s - loss: 7.6832 - accuracy: 0.4989
10208/25000 [===========>..................] - ETA: 35s - loss: 7.6846 - accuracy: 0.4988
10240/25000 [===========>..................] - ETA: 35s - loss: 7.6861 - accuracy: 0.4987
10272/25000 [===========>..................] - ETA: 35s - loss: 7.6890 - accuracy: 0.4985
10304/25000 [===========>..................] - ETA: 35s - loss: 7.6934 - accuracy: 0.4983
10336/25000 [===========>..................] - ETA: 35s - loss: 7.6978 - accuracy: 0.4980
10368/25000 [===========>..................] - ETA: 35s - loss: 7.7006 - accuracy: 0.4978
10400/25000 [===========>..................] - ETA: 35s - loss: 7.6946 - accuracy: 0.4982
10432/25000 [===========>..................] - ETA: 35s - loss: 7.7004 - accuracy: 0.4978
10464/25000 [===========>..................] - ETA: 35s - loss: 7.7018 - accuracy: 0.4977
10496/25000 [===========>..................] - ETA: 35s - loss: 7.6929 - accuracy: 0.4983
10528/25000 [===========>..................] - ETA: 35s - loss: 7.6943 - accuracy: 0.4982
10560/25000 [===========>..................] - ETA: 35s - loss: 7.6899 - accuracy: 0.4985
10592/25000 [===========>..................] - ETA: 35s - loss: 7.6898 - accuracy: 0.4985
10624/25000 [===========>..................] - ETA: 34s - loss: 7.6940 - accuracy: 0.4982
10656/25000 [===========>..................] - ETA: 34s - loss: 7.6925 - accuracy: 0.4983
10688/25000 [===========>..................] - ETA: 34s - loss: 7.6910 - accuracy: 0.4984
10720/25000 [===========>..................] - ETA: 34s - loss: 7.6952 - accuracy: 0.4981
10752/25000 [===========>..................] - ETA: 34s - loss: 7.6966 - accuracy: 0.4980
10784/25000 [===========>..................] - ETA: 34s - loss: 7.7022 - accuracy: 0.4977
10816/25000 [===========>..................] - ETA: 34s - loss: 7.6964 - accuracy: 0.4981
10848/25000 [============>.................] - ETA: 34s - loss: 7.6935 - accuracy: 0.4982
10880/25000 [============>.................] - ETA: 34s - loss: 7.6934 - accuracy: 0.4983
10912/25000 [============>.................] - ETA: 34s - loss: 7.6975 - accuracy: 0.4980
10944/25000 [============>.................] - ETA: 34s - loss: 7.6904 - accuracy: 0.4984
10976/25000 [============>.................] - ETA: 34s - loss: 7.6890 - accuracy: 0.4985
11008/25000 [============>.................] - ETA: 33s - loss: 7.6945 - accuracy: 0.4982
11040/25000 [============>.................] - ETA: 33s - loss: 7.6958 - accuracy: 0.4981
11072/25000 [============>.................] - ETA: 33s - loss: 7.7026 - accuracy: 0.4977
11104/25000 [============>.................] - ETA: 33s - loss: 7.7053 - accuracy: 0.4975
11136/25000 [============>.................] - ETA: 33s - loss: 7.7052 - accuracy: 0.4975
11168/25000 [============>.................] - ETA: 33s - loss: 7.7078 - accuracy: 0.4973
11200/25000 [============>.................] - ETA: 33s - loss: 7.7036 - accuracy: 0.4976
11232/25000 [============>.................] - ETA: 33s - loss: 7.7048 - accuracy: 0.4975
11264/25000 [============>.................] - ETA: 33s - loss: 7.7047 - accuracy: 0.4975
11296/25000 [============>.................] - ETA: 33s - loss: 7.7060 - accuracy: 0.4974
11328/25000 [============>.................] - ETA: 33s - loss: 7.7005 - accuracy: 0.4978
11360/25000 [============>.................] - ETA: 33s - loss: 7.6977 - accuracy: 0.4980
11392/25000 [============>.................] - ETA: 33s - loss: 7.6908 - accuracy: 0.4984
11424/25000 [============>.................] - ETA: 32s - loss: 7.6881 - accuracy: 0.4986
11456/25000 [============>.................] - ETA: 32s - loss: 7.6867 - accuracy: 0.4987
11488/25000 [============>.................] - ETA: 32s - loss: 7.6880 - accuracy: 0.4986
11520/25000 [============>.................] - ETA: 32s - loss: 7.6866 - accuracy: 0.4987
11552/25000 [============>.................] - ETA: 32s - loss: 7.6812 - accuracy: 0.4990
11584/25000 [============>.................] - ETA: 32s - loss: 7.6852 - accuracy: 0.4988
11616/25000 [============>.................] - ETA: 32s - loss: 7.6825 - accuracy: 0.4990
11648/25000 [============>.................] - ETA: 32s - loss: 7.6798 - accuracy: 0.4991
11680/25000 [=============>................] - ETA: 32s - loss: 7.6732 - accuracy: 0.4996
11712/25000 [=============>................] - ETA: 32s - loss: 7.6719 - accuracy: 0.4997
11744/25000 [=============>................] - ETA: 32s - loss: 7.6731 - accuracy: 0.4996
11776/25000 [=============>................] - ETA: 32s - loss: 7.6692 - accuracy: 0.4998
11808/25000 [=============>................] - ETA: 32s - loss: 7.6744 - accuracy: 0.4995
11840/25000 [=============>................] - ETA: 31s - loss: 7.6744 - accuracy: 0.4995
11872/25000 [=============>................] - ETA: 31s - loss: 7.6705 - accuracy: 0.4997
11904/25000 [=============>................] - ETA: 31s - loss: 7.6731 - accuracy: 0.4996
11936/25000 [=============>................] - ETA: 31s - loss: 7.6782 - accuracy: 0.4992
11968/25000 [=============>................] - ETA: 31s - loss: 7.6782 - accuracy: 0.4992
12000/25000 [=============>................] - ETA: 31s - loss: 7.6743 - accuracy: 0.4995
12032/25000 [=============>................] - ETA: 31s - loss: 7.6755 - accuracy: 0.4994
12064/25000 [=============>................] - ETA: 31s - loss: 7.6742 - accuracy: 0.4995
12096/25000 [=============>................] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
12128/25000 [=============>................] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
12160/25000 [=============>................] - ETA: 31s - loss: 7.6691 - accuracy: 0.4998
12192/25000 [=============>................] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
12224/25000 [=============>................] - ETA: 31s - loss: 7.6654 - accuracy: 0.5001
12256/25000 [=============>................] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
12288/25000 [=============>................] - ETA: 30s - loss: 7.6679 - accuracy: 0.4999
12320/25000 [=============>................] - ETA: 30s - loss: 7.6741 - accuracy: 0.4995
12352/25000 [=============>................] - ETA: 30s - loss: 7.6741 - accuracy: 0.4995
12384/25000 [=============>................] - ETA: 30s - loss: 7.6753 - accuracy: 0.4994
12416/25000 [=============>................] - ETA: 30s - loss: 7.6716 - accuracy: 0.4997
12448/25000 [=============>................] - ETA: 30s - loss: 7.6691 - accuracy: 0.4998
12480/25000 [=============>................] - ETA: 30s - loss: 7.6703 - accuracy: 0.4998
12512/25000 [==============>...............] - ETA: 30s - loss: 7.6654 - accuracy: 0.5001
12544/25000 [==============>...............] - ETA: 30s - loss: 7.6642 - accuracy: 0.5002
12576/25000 [==============>...............] - ETA: 30s - loss: 7.6642 - accuracy: 0.5002
12608/25000 [==============>...............] - ETA: 30s - loss: 7.6642 - accuracy: 0.5002
12640/25000 [==============>...............] - ETA: 30s - loss: 7.6581 - accuracy: 0.5006
12672/25000 [==============>...............] - ETA: 29s - loss: 7.6594 - accuracy: 0.5005
12704/25000 [==============>...............] - ETA: 29s - loss: 7.6582 - accuracy: 0.5006
12736/25000 [==============>...............] - ETA: 29s - loss: 7.6558 - accuracy: 0.5007
12768/25000 [==============>...............] - ETA: 29s - loss: 7.6558 - accuracy: 0.5007
12800/25000 [==============>...............] - ETA: 29s - loss: 7.6498 - accuracy: 0.5011
12832/25000 [==============>...............] - ETA: 29s - loss: 7.6523 - accuracy: 0.5009
12864/25000 [==============>...............] - ETA: 29s - loss: 7.6511 - accuracy: 0.5010
12896/25000 [==============>...............] - ETA: 29s - loss: 7.6428 - accuracy: 0.5016
12928/25000 [==============>...............] - ETA: 29s - loss: 7.6405 - accuracy: 0.5017
12960/25000 [==============>...............] - ETA: 29s - loss: 7.6477 - accuracy: 0.5012
12992/25000 [==============>...............] - ETA: 29s - loss: 7.6430 - accuracy: 0.5015
13024/25000 [==============>...............] - ETA: 29s - loss: 7.6478 - accuracy: 0.5012
13056/25000 [==============>...............] - ETA: 28s - loss: 7.6455 - accuracy: 0.5014
13088/25000 [==============>...............] - ETA: 28s - loss: 7.6432 - accuracy: 0.5015
13120/25000 [==============>...............] - ETA: 28s - loss: 7.6444 - accuracy: 0.5014
13152/25000 [==============>...............] - ETA: 28s - loss: 7.6410 - accuracy: 0.5017
13184/25000 [==============>...............] - ETA: 28s - loss: 7.6445 - accuracy: 0.5014
13216/25000 [==============>...............] - ETA: 28s - loss: 7.6423 - accuracy: 0.5016
13248/25000 [==============>...............] - ETA: 28s - loss: 7.6412 - accuracy: 0.5017
13280/25000 [==============>...............] - ETA: 28s - loss: 7.6366 - accuracy: 0.5020
13312/25000 [==============>...............] - ETA: 28s - loss: 7.6332 - accuracy: 0.5022
13344/25000 [===============>..............] - ETA: 28s - loss: 7.6379 - accuracy: 0.5019
13376/25000 [===============>..............] - ETA: 28s - loss: 7.6403 - accuracy: 0.5017
13408/25000 [===============>..............] - ETA: 28s - loss: 7.6426 - accuracy: 0.5016
13440/25000 [===============>..............] - ETA: 28s - loss: 7.6415 - accuracy: 0.5016
13472/25000 [===============>..............] - ETA: 27s - loss: 7.6427 - accuracy: 0.5016
13504/25000 [===============>..............] - ETA: 27s - loss: 7.6428 - accuracy: 0.5016
13536/25000 [===============>..............] - ETA: 27s - loss: 7.6474 - accuracy: 0.5013
13568/25000 [===============>..............] - ETA: 27s - loss: 7.6429 - accuracy: 0.5015
13600/25000 [===============>..............] - ETA: 27s - loss: 7.6429 - accuracy: 0.5015
13632/25000 [===============>..............] - ETA: 27s - loss: 7.6509 - accuracy: 0.5010
13664/25000 [===============>..............] - ETA: 27s - loss: 7.6621 - accuracy: 0.5003
13696/25000 [===============>..............] - ETA: 27s - loss: 7.6700 - accuracy: 0.4998
13728/25000 [===============>..............] - ETA: 27s - loss: 7.6744 - accuracy: 0.4995
13760/25000 [===============>..............] - ETA: 27s - loss: 7.6744 - accuracy: 0.4995
13792/25000 [===============>..............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
13824/25000 [===============>..............] - ETA: 27s - loss: 7.6655 - accuracy: 0.5001
13856/25000 [===============>..............] - ETA: 27s - loss: 7.6644 - accuracy: 0.5001
13888/25000 [===============>..............] - ETA: 26s - loss: 7.6633 - accuracy: 0.5002
13920/25000 [===============>..............] - ETA: 26s - loss: 7.6622 - accuracy: 0.5003
13952/25000 [===============>..............] - ETA: 26s - loss: 7.6589 - accuracy: 0.5005
13984/25000 [===============>..............] - ETA: 26s - loss: 7.6589 - accuracy: 0.5005
14016/25000 [===============>..............] - ETA: 26s - loss: 7.6601 - accuracy: 0.5004
14048/25000 [===============>..............] - ETA: 26s - loss: 7.6601 - accuracy: 0.5004
14080/25000 [===============>..............] - ETA: 26s - loss: 7.6623 - accuracy: 0.5003
14112/25000 [===============>..............] - ETA: 26s - loss: 7.6612 - accuracy: 0.5004
14144/25000 [===============>..............] - ETA: 26s - loss: 7.6623 - accuracy: 0.5003
14176/25000 [================>.............] - ETA: 26s - loss: 7.6655 - accuracy: 0.5001
14208/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14240/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14272/25000 [================>.............] - ETA: 26s - loss: 7.6688 - accuracy: 0.4999
14304/25000 [================>.............] - ETA: 25s - loss: 7.6698 - accuracy: 0.4998
14336/25000 [================>.............] - ETA: 25s - loss: 7.6762 - accuracy: 0.4994
14368/25000 [================>.............] - ETA: 25s - loss: 7.6762 - accuracy: 0.4994
14400/25000 [================>.............] - ETA: 25s - loss: 7.6783 - accuracy: 0.4992
14432/25000 [================>.............] - ETA: 25s - loss: 7.6751 - accuracy: 0.4994
14464/25000 [================>.............] - ETA: 25s - loss: 7.6709 - accuracy: 0.4997
14496/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
14528/25000 [================>.............] - ETA: 25s - loss: 7.6698 - accuracy: 0.4998
14560/25000 [================>.............] - ETA: 25s - loss: 7.6750 - accuracy: 0.4995
14592/25000 [================>.............] - ETA: 25s - loss: 7.6719 - accuracy: 0.4997
14624/25000 [================>.............] - ETA: 25s - loss: 7.6719 - accuracy: 0.4997
14656/25000 [================>.............] - ETA: 25s - loss: 7.6750 - accuracy: 0.4995
14688/25000 [================>.............] - ETA: 24s - loss: 7.6729 - accuracy: 0.4996
14720/25000 [================>.............] - ETA: 24s - loss: 7.6677 - accuracy: 0.4999
14752/25000 [================>.............] - ETA: 24s - loss: 7.6677 - accuracy: 0.4999
14784/25000 [================>.............] - ETA: 24s - loss: 7.6677 - accuracy: 0.4999
14816/25000 [================>.............] - ETA: 24s - loss: 7.6677 - accuracy: 0.4999
14848/25000 [================>.............] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
14880/25000 [================>.............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
14912/25000 [================>.............] - ETA: 24s - loss: 7.6697 - accuracy: 0.4998
14944/25000 [================>.............] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
14976/25000 [================>.............] - ETA: 24s - loss: 7.6748 - accuracy: 0.4995
15008/25000 [=================>............] - ETA: 24s - loss: 7.6727 - accuracy: 0.4996
15040/25000 [=================>............] - ETA: 24s - loss: 7.6697 - accuracy: 0.4998
15072/25000 [=================>............] - ETA: 24s - loss: 7.6676 - accuracy: 0.4999
15104/25000 [=================>............] - ETA: 23s - loss: 7.6656 - accuracy: 0.5001
15136/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15168/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15200/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15232/25000 [=================>............] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
15264/25000 [=================>............] - ETA: 23s - loss: 7.6696 - accuracy: 0.4998
15296/25000 [=================>............] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
15328/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15360/25000 [=================>............] - ETA: 23s - loss: 7.6636 - accuracy: 0.5002
15392/25000 [=================>............] - ETA: 23s - loss: 7.6656 - accuracy: 0.5001
15424/25000 [=================>............] - ETA: 23s - loss: 7.6616 - accuracy: 0.5003
15456/25000 [=================>............] - ETA: 23s - loss: 7.6636 - accuracy: 0.5002
15488/25000 [=================>............] - ETA: 23s - loss: 7.6656 - accuracy: 0.5001
15520/25000 [=================>............] - ETA: 22s - loss: 7.6607 - accuracy: 0.5004
15552/25000 [=================>............] - ETA: 22s - loss: 7.6587 - accuracy: 0.5005
15584/25000 [=================>............] - ETA: 22s - loss: 7.6597 - accuracy: 0.5004
15616/25000 [=================>............] - ETA: 22s - loss: 7.6617 - accuracy: 0.5003
15648/25000 [=================>............] - ETA: 22s - loss: 7.6578 - accuracy: 0.5006
15680/25000 [=================>............] - ETA: 22s - loss: 7.6598 - accuracy: 0.5004
15712/25000 [=================>............] - ETA: 22s - loss: 7.6598 - accuracy: 0.5004
15744/25000 [=================>............] - ETA: 22s - loss: 7.6637 - accuracy: 0.5002
15776/25000 [=================>............] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15808/25000 [=================>............] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
15840/25000 [==================>...........] - ETA: 22s - loss: 7.6647 - accuracy: 0.5001
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15936/25000 [==================>...........] - ETA: 21s - loss: 7.6637 - accuracy: 0.5002
15968/25000 [==================>...........] - ETA: 21s - loss: 7.6618 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 21s - loss: 7.6590 - accuracy: 0.5005
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6599 - accuracy: 0.5004
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6618 - accuracy: 0.5003
16096/25000 [==================>...........] - ETA: 21s - loss: 7.6590 - accuracy: 0.5005
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6590 - accuracy: 0.5005
16160/25000 [==================>...........] - ETA: 21s - loss: 7.6600 - accuracy: 0.5004
16192/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16224/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6713 - accuracy: 0.4997
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6713 - accuracy: 0.4997
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6704 - accuracy: 0.4998
16352/25000 [==================>...........] - ETA: 20s - loss: 7.6685 - accuracy: 0.4999
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6638 - accuracy: 0.5002
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6620 - accuracy: 0.5003
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6638 - accuracy: 0.5002
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6657 - accuracy: 0.5001
16544/25000 [==================>...........] - ETA: 20s - loss: 7.6648 - accuracy: 0.5001
16576/25000 [==================>...........] - ETA: 20s - loss: 7.6611 - accuracy: 0.5004
16608/25000 [==================>...........] - ETA: 20s - loss: 7.6638 - accuracy: 0.5002
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6611 - accuracy: 0.5004
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6657 - accuracy: 0.5001
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6694 - accuracy: 0.4998
16736/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
16768/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6630 - accuracy: 0.5002
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
16960/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
16992/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6603 - accuracy: 0.5004
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6594 - accuracy: 0.5005
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6568 - accuracy: 0.5006
17152/25000 [===================>..........] - ETA: 18s - loss: 7.6604 - accuracy: 0.5004
17184/25000 [===================>..........] - ETA: 18s - loss: 7.6550 - accuracy: 0.5008
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6550 - accuracy: 0.5008
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6551 - accuracy: 0.5008
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6542 - accuracy: 0.5008
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6516 - accuracy: 0.5010
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6472 - accuracy: 0.5013
17376/25000 [===================>..........] - ETA: 18s - loss: 7.6499 - accuracy: 0.5011
17408/25000 [===================>..........] - ETA: 18s - loss: 7.6472 - accuracy: 0.5013
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6482 - accuracy: 0.5012
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6491 - accuracy: 0.5011
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6465 - accuracy: 0.5013
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6456 - accuracy: 0.5014
17568/25000 [====================>.........] - ETA: 17s - loss: 7.6500 - accuracy: 0.5011
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6501 - accuracy: 0.5011
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6536 - accuracy: 0.5009
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6562 - accuracy: 0.5007
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6580 - accuracy: 0.5006
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6588 - accuracy: 0.5005
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6563 - accuracy: 0.5007
17792/25000 [====================>.........] - ETA: 17s - loss: 7.6563 - accuracy: 0.5007
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6563 - accuracy: 0.5007
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6555 - accuracy: 0.5007
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6538 - accuracy: 0.5008
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6546 - accuracy: 0.5008
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6495 - accuracy: 0.5011
17984/25000 [====================>.........] - ETA: 16s - loss: 7.6513 - accuracy: 0.5010
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6573 - accuracy: 0.5006
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6590 - accuracy: 0.5005
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6632 - accuracy: 0.5002
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6675 - accuracy: 0.4999
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6658 - accuracy: 0.5001
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6633 - accuracy: 0.5002
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6599 - accuracy: 0.5004
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6565 - accuracy: 0.5007
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6582 - accuracy: 0.5005
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6608 - accuracy: 0.5004
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6583 - accuracy: 0.5005
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6600 - accuracy: 0.5004
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6567 - accuracy: 0.5006
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6575 - accuracy: 0.5006
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6641 - accuracy: 0.5002
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6699 - accuracy: 0.4998
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6716 - accuracy: 0.4997
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6732 - accuracy: 0.4996
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6740 - accuracy: 0.4995
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6748 - accuracy: 0.4995
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6740 - accuracy: 0.4995
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6715 - accuracy: 0.4997
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6699 - accuracy: 0.4998
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6723 - accuracy: 0.4996
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6699 - accuracy: 0.4998
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6699 - accuracy: 0.4998
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6674 - accuracy: 0.4999
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6658 - accuracy: 0.5001
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6650 - accuracy: 0.5001
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6682 - accuracy: 0.4999
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6698 - accuracy: 0.4998
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6698 - accuracy: 0.4998
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6706 - accuracy: 0.4997
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6706 - accuracy: 0.4997
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6721 - accuracy: 0.4996
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6713 - accuracy: 0.4997
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6705 - accuracy: 0.4997
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6705 - accuracy: 0.4997
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6728 - accuracy: 0.4996
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6713 - accuracy: 0.4997
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6712 - accuracy: 0.4997
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6712 - accuracy: 0.4997
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6712 - accuracy: 0.4997
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6705 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6720 - accuracy: 0.4997
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6727 - accuracy: 0.4996
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6735 - accuracy: 0.4996
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6727 - accuracy: 0.4996
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6689 - accuracy: 0.4999
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6697 - accuracy: 0.4998
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6689 - accuracy: 0.4999
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6651 - accuracy: 0.5001
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6628 - accuracy: 0.5002
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6613 - accuracy: 0.5003
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6651 - accuracy: 0.5001
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6689 - accuracy: 0.4999
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6704 - accuracy: 0.4998
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6726 - accuracy: 0.4996
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6711 - accuracy: 0.4997
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6771 - accuracy: 0.4993
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6726 - accuracy: 0.4996
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6711 - accuracy: 0.4997
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6711 - accuracy: 0.4997
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6711 - accuracy: 0.4997
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6674 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6674 - accuracy: 0.5000
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999 
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6710 - accuracy: 0.4997
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6739 - accuracy: 0.4995
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6747 - accuracy: 0.4995
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6717 - accuracy: 0.4997
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6688 - accuracy: 0.4999
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6681 - accuracy: 0.4999
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6681 - accuracy: 0.4999
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6716 - accuracy: 0.4997
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6781 - accuracy: 0.4993
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6816 - accuracy: 0.4990
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6809 - accuracy: 0.4991
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6830 - accuracy: 0.4989
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6823 - accuracy: 0.4990
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6822 - accuracy: 0.4990
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6892 - accuracy: 0.4985
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6885 - accuracy: 0.4986
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6892 - accuracy: 0.4985
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6891 - accuracy: 0.4985
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6884 - accuracy: 0.4986
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6884 - accuracy: 0.4986
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6897 - accuracy: 0.4985
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6911 - accuracy: 0.4984
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6946 - accuracy: 0.4982
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6959 - accuracy: 0.4981
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6952 - accuracy: 0.4981
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6972 - accuracy: 0.4980
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6986 - accuracy: 0.4979
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6992 - accuracy: 0.4979
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6971 - accuracy: 0.4980
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6950 - accuracy: 0.4982
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6935 - accuracy: 0.4982
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6921 - accuracy: 0.4983
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6893 - accuracy: 0.4985
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6893 - accuracy: 0.4985
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6927 - accuracy: 0.4983
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6940 - accuracy: 0.4982
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6926 - accuracy: 0.4983
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6933 - accuracy: 0.4983
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6939 - accuracy: 0.4982
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6918 - accuracy: 0.4984
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6898 - accuracy: 0.4985
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6870 - accuracy: 0.4987
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6877 - accuracy: 0.4986
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6876 - accuracy: 0.4986
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6862 - accuracy: 0.4987
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6869 - accuracy: 0.4987
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6848 - accuracy: 0.4988
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6848 - accuracy: 0.4988
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6814 - accuracy: 0.4990
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6807 - accuracy: 0.4991
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6814 - accuracy: 0.4990
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6787 - accuracy: 0.4992
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6813 - accuracy: 0.4990
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6780 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6740 - accuracy: 0.4995
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6740 - accuracy: 0.4995
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6786 - accuracy: 0.4992
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6766 - accuracy: 0.4993
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6772 - accuracy: 0.4993
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6739 - accuracy: 0.4995
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6759 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6785 - accuracy: 0.4992
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6752 - accuracy: 0.4994
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6765 - accuracy: 0.4994
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6778 - accuracy: 0.4993
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6745 - accuracy: 0.4995
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6732 - accuracy: 0.4996
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6777 - accuracy: 0.4993
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6757 - accuracy: 0.4994
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24192/25000 [============================>.] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
24224/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
24256/25000 [============================>.] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
24288/25000 [============================>.] - ETA: 1s - loss: 7.6546 - accuracy: 0.5008
24320/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24352/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24384/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24416/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
24448/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24480/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24512/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24544/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24576/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24608/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24768/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24800/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24832/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 71s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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


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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:53<01:20, 26.68s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.13771447836750392, 'embedding_size_factor': 1.012870274360068, 'layers.choice': 3, 'learning_rate': 0.004140106140835842, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 6.119004816426828e-09} and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc1\xa0\xa0\xc6c\x16\xa7X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf04\xb7u\xf7x\xfeX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?p\xf57G\xa16\x9aX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>:G\xea\xbc\xfak\xfbu.' and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc1\xa0\xa0\xc6c\x16\xa7X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf04\xb7u\xf7x\xfeX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?p\xf57G\xa16\x9aX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>:G\xea\xbc\xfak\xfbu.' and reward: 0.3912
 60%|██████    | 3/5 [01:47<01:10, 35.06s/it] 60%|██████    | 3/5 [01:47<01:11, 35.99s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4260260561596832, 'embedding_size_factor': 1.3332725452428167, 'layers.choice': 1, 'learning_rate': 0.0039783446838092965, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 5.735465705505283e-05} and reward: 0.3754
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdbD\x02\xca\x9c\xc7\xddX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5U\x15\x97\xa7\x8f\xaeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?pK\x98\xc4\xf2b\xf1X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x0e\x12\x03\x02\xb6\x8c\x9cu.' and reward: 0.3754
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdbD\x02\xca\x9c\xc7\xddX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5U\x15\x97\xa7\x8f\xaeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?pK\x98\xc4\xf2b\xf1X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x0e\x12\x03\x02\xb6\x8c\x9cu.' and reward: 0.3754
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 208.01020169258118
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.13771447836750392, 'embedding_size_factor': 1.012870274360068, 'layers.choice': 3, 'learning_rate': 0.004140106140835842, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 6.119004816426828e-09}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -90.79s of remaining time.
Ensemble size: 55
Ensemble weights: 
[0.50909091 0.49090909 0.        ]
	0.3954	 = Validation accuracy score
	1.1s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 211.93s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl





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
 1449984/17464789 [=>............................] - ETA: 0s
 5488640/17464789 [========>.....................] - ETA: 0s
11911168/17464789 [===================>..........] - ETA: 0s
17047552/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-24 03:43:21.507340: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-24 03:43:21.511586: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-24 03:43:21.511758: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563b5662f4c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 03:43:21.511773: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:58 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 3:04 - loss: 5.2708 - accuracy: 0.6562
   96/25000 [..............................] - ETA: 2:24 - loss: 6.5486 - accuracy: 0.5729
  128/25000 [..............................] - ETA: 2:04 - loss: 6.5885 - accuracy: 0.5703
  160/25000 [..............................] - ETA: 1:53 - loss: 6.9958 - accuracy: 0.5437
  192/25000 [..............................] - ETA: 1:45 - loss: 7.1875 - accuracy: 0.5312
  224/25000 [..............................] - ETA: 1:40 - loss: 7.2559 - accuracy: 0.5268
  256/25000 [..............................] - ETA: 1:36 - loss: 7.2474 - accuracy: 0.5273
  288/25000 [..............................] - ETA: 1:33 - loss: 7.4537 - accuracy: 0.5139
  320/25000 [..............................] - ETA: 1:30 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:28 - loss: 7.4488 - accuracy: 0.5142
  384/25000 [..............................] - ETA: 1:26 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:25 - loss: 7.4086 - accuracy: 0.5168
  448/25000 [..............................] - ETA: 1:24 - loss: 7.3928 - accuracy: 0.5179
  480/25000 [..............................] - ETA: 1:23 - loss: 7.3791 - accuracy: 0.5188
  512/25000 [..............................] - ETA: 1:22 - loss: 7.5468 - accuracy: 0.5078
  544/25000 [..............................] - ETA: 1:21 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 1:20 - loss: 7.5335 - accuracy: 0.5087
  608/25000 [..............................] - ETA: 1:19 - loss: 7.3892 - accuracy: 0.5181
  640/25000 [..............................] - ETA: 1:18 - loss: 7.4750 - accuracy: 0.5125
  672/25000 [..............................] - ETA: 1:18 - loss: 7.4841 - accuracy: 0.5119
  704/25000 [..............................] - ETA: 1:17 - loss: 7.4706 - accuracy: 0.5128
  736/25000 [..............................] - ETA: 1:17 - loss: 7.4375 - accuracy: 0.5149
  768/25000 [..............................] - ETA: 1:16 - loss: 7.4270 - accuracy: 0.5156
  800/25000 [..............................] - ETA: 1:16 - loss: 7.3791 - accuracy: 0.5188
  832/25000 [..............................] - ETA: 1:16 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:15 - loss: 7.3827 - accuracy: 0.5185
  896/25000 [>.............................] - ETA: 1:15 - loss: 7.3586 - accuracy: 0.5201
  928/25000 [>.............................] - ETA: 1:15 - loss: 7.4518 - accuracy: 0.5140
  960/25000 [>.............................] - ETA: 1:14 - loss: 7.4750 - accuracy: 0.5125
  992/25000 [>.............................] - ETA: 1:14 - loss: 7.4502 - accuracy: 0.5141
 1024/25000 [>.............................] - ETA: 1:14 - loss: 7.5468 - accuracy: 0.5078
 1056/25000 [>.............................] - ETA: 1:13 - loss: 7.5650 - accuracy: 0.5066
 1088/25000 [>.............................] - ETA: 1:13 - loss: 7.5962 - accuracy: 0.5046
 1120/25000 [>.............................] - ETA: 1:13 - loss: 7.5571 - accuracy: 0.5071
 1152/25000 [>.............................] - ETA: 1:12 - loss: 7.6267 - accuracy: 0.5026
 1184/25000 [>.............................] - ETA: 1:12 - loss: 7.5760 - accuracy: 0.5059
 1216/25000 [>.............................] - ETA: 1:12 - loss: 7.5405 - accuracy: 0.5082
 1248/25000 [>.............................] - ETA: 1:12 - loss: 7.5683 - accuracy: 0.5064
 1280/25000 [>.............................] - ETA: 1:11 - loss: 7.5468 - accuracy: 0.5078
 1312/25000 [>.............................] - ETA: 1:11 - loss: 7.5264 - accuracy: 0.5091
 1344/25000 [>.............................] - ETA: 1:11 - loss: 7.5411 - accuracy: 0.5082
 1376/25000 [>.............................] - ETA: 1:10 - loss: 7.5886 - accuracy: 0.5051
 1408/25000 [>.............................] - ETA: 1:10 - loss: 7.5795 - accuracy: 0.5057
 1440/25000 [>.............................] - ETA: 1:10 - loss: 7.5495 - accuracy: 0.5076
 1472/25000 [>.............................] - ETA: 1:10 - loss: 7.6041 - accuracy: 0.5041
 1504/25000 [>.............................] - ETA: 1:10 - loss: 7.5953 - accuracy: 0.5047
 1536/25000 [>.............................] - ETA: 1:09 - loss: 7.6267 - accuracy: 0.5026
 1568/25000 [>.............................] - ETA: 1:09 - loss: 7.6079 - accuracy: 0.5038
 1600/25000 [>.............................] - ETA: 1:09 - loss: 7.6091 - accuracy: 0.5038
 1632/25000 [>.............................] - ETA: 1:09 - loss: 7.6196 - accuracy: 0.5031
 1664/25000 [>.............................] - ETA: 1:09 - loss: 7.6021 - accuracy: 0.5042
 1696/25000 [=>............................] - ETA: 1:08 - loss: 7.6033 - accuracy: 0.5041
 1728/25000 [=>............................] - ETA: 1:08 - loss: 7.5601 - accuracy: 0.5069
 1760/25000 [=>............................] - ETA: 1:08 - loss: 7.5708 - accuracy: 0.5063
 1792/25000 [=>............................] - ETA: 1:08 - loss: 7.5639 - accuracy: 0.5067
 1824/25000 [=>............................] - ETA: 1:08 - loss: 7.5573 - accuracy: 0.5071
 1856/25000 [=>............................] - ETA: 1:07 - loss: 7.5014 - accuracy: 0.5108
 1888/25000 [=>............................] - ETA: 1:07 - loss: 7.5123 - accuracy: 0.5101
 1920/25000 [=>............................] - ETA: 1:07 - loss: 7.5149 - accuracy: 0.5099
 1952/25000 [=>............................] - ETA: 1:07 - loss: 7.5566 - accuracy: 0.5072
 1984/25000 [=>............................] - ETA: 1:07 - loss: 7.5661 - accuracy: 0.5066
 2016/25000 [=>............................] - ETA: 1:07 - loss: 7.6058 - accuracy: 0.5040
 2048/25000 [=>............................] - ETA: 1:06 - loss: 7.5918 - accuracy: 0.5049
 2080/25000 [=>............................] - ETA: 1:06 - loss: 7.5782 - accuracy: 0.5058
 2112/25000 [=>............................] - ETA: 1:06 - loss: 7.5650 - accuracy: 0.5066
 2144/25000 [=>............................] - ETA: 1:06 - loss: 7.5450 - accuracy: 0.5079
 2176/25000 [=>............................] - ETA: 1:06 - loss: 7.5327 - accuracy: 0.5087
 2208/25000 [=>............................] - ETA: 1:06 - loss: 7.5277 - accuracy: 0.5091
 2240/25000 [=>............................] - ETA: 1:05 - loss: 7.5366 - accuracy: 0.5085
 2272/25000 [=>............................] - ETA: 1:05 - loss: 7.5114 - accuracy: 0.5101
 2304/25000 [=>............................] - ETA: 1:05 - loss: 7.5069 - accuracy: 0.5104
 2336/25000 [=>............................] - ETA: 1:05 - loss: 7.5025 - accuracy: 0.5107
 2368/25000 [=>............................] - ETA: 1:05 - loss: 7.4853 - accuracy: 0.5118
 2400/25000 [=>............................] - ETA: 1:05 - loss: 7.4813 - accuracy: 0.5121
 2432/25000 [=>............................] - ETA: 1:04 - loss: 7.4964 - accuracy: 0.5111
 2464/25000 [=>............................] - ETA: 1:04 - loss: 7.5110 - accuracy: 0.5101
 2496/25000 [=>............................] - ETA: 1:04 - loss: 7.5253 - accuracy: 0.5092
 2528/25000 [==>...........................] - ETA: 1:04 - loss: 7.5332 - accuracy: 0.5087
 2560/25000 [==>...........................] - ETA: 1:04 - loss: 7.5109 - accuracy: 0.5102
 2592/25000 [==>...........................] - ETA: 1:04 - loss: 7.5483 - accuracy: 0.5077
 2624/25000 [==>...........................] - ETA: 1:03 - loss: 7.5673 - accuracy: 0.5065
 2656/25000 [==>...........................] - ETA: 1:03 - loss: 7.5627 - accuracy: 0.5068
 2688/25000 [==>...........................] - ETA: 1:03 - loss: 7.5354 - accuracy: 0.5086
 2720/25000 [==>...........................] - ETA: 1:03 - loss: 7.5313 - accuracy: 0.5088
 2752/25000 [==>...........................] - ETA: 1:03 - loss: 7.5440 - accuracy: 0.5080
 2784/25000 [==>...........................] - ETA: 1:03 - loss: 7.5565 - accuracy: 0.5072
 2816/25000 [==>...........................] - ETA: 1:03 - loss: 7.5468 - accuracy: 0.5078
 2848/25000 [==>...........................] - ETA: 1:03 - loss: 7.5428 - accuracy: 0.5081
 2880/25000 [==>...........................] - ETA: 1:03 - loss: 7.5282 - accuracy: 0.5090
 2912/25000 [==>...........................] - ETA: 1:02 - loss: 7.5350 - accuracy: 0.5086
 2944/25000 [==>...........................] - ETA: 1:02 - loss: 7.5364 - accuracy: 0.5085
 2976/25000 [==>...........................] - ETA: 1:02 - loss: 7.5481 - accuracy: 0.5077
 3008/25000 [==>...........................] - ETA: 1:02 - loss: 7.5443 - accuracy: 0.5080
 3040/25000 [==>...........................] - ETA: 1:02 - loss: 7.5456 - accuracy: 0.5079
 3072/25000 [==>...........................] - ETA: 1:02 - loss: 7.5368 - accuracy: 0.5085
 3104/25000 [==>...........................] - ETA: 1:02 - loss: 7.5332 - accuracy: 0.5087
 3136/25000 [==>...........................] - ETA: 1:02 - loss: 7.5444 - accuracy: 0.5080
 3168/25000 [==>...........................] - ETA: 1:02 - loss: 7.5843 - accuracy: 0.5054
 3200/25000 [==>...........................] - ETA: 1:01 - loss: 7.5900 - accuracy: 0.5050
 3232/25000 [==>...........................] - ETA: 1:01 - loss: 7.6049 - accuracy: 0.5040
 3264/25000 [==>...........................] - ETA: 1:01 - loss: 7.6149 - accuracy: 0.5034
 3296/25000 [==>...........................] - ETA: 1:01 - loss: 7.6061 - accuracy: 0.5039
 3328/25000 [==>...........................] - ETA: 1:01 - loss: 7.5929 - accuracy: 0.5048
 3360/25000 [===>..........................] - ETA: 1:01 - loss: 7.5845 - accuracy: 0.5054
 3392/25000 [===>..........................] - ETA: 1:01 - loss: 7.5943 - accuracy: 0.5047
 3424/25000 [===>..........................] - ETA: 1:00 - loss: 7.5905 - accuracy: 0.5050
 3456/25000 [===>..........................] - ETA: 1:00 - loss: 7.6045 - accuracy: 0.5041
 3488/25000 [===>..........................] - ETA: 1:00 - loss: 7.6183 - accuracy: 0.5032
 3520/25000 [===>..........................] - ETA: 1:00 - loss: 7.6318 - accuracy: 0.5023
 3552/25000 [===>..........................] - ETA: 1:00 - loss: 7.6407 - accuracy: 0.5017
 3584/25000 [===>..........................] - ETA: 1:00 - loss: 7.6452 - accuracy: 0.5014
 3616/25000 [===>..........................] - ETA: 1:00 - loss: 7.6454 - accuracy: 0.5014
 3648/25000 [===>..........................] - ETA: 1:00 - loss: 7.6456 - accuracy: 0.5014
 3680/25000 [===>..........................] - ETA: 59s - loss: 7.6500 - accuracy: 0.5011 
 3712/25000 [===>..........................] - ETA: 59s - loss: 7.6501 - accuracy: 0.5011
 3744/25000 [===>..........................] - ETA: 59s - loss: 7.6543 - accuracy: 0.5008
 3776/25000 [===>..........................] - ETA: 59s - loss: 7.6747 - accuracy: 0.4995
 3808/25000 [===>..........................] - ETA: 59s - loss: 7.6787 - accuracy: 0.4992
 3840/25000 [===>..........................] - ETA: 59s - loss: 7.6866 - accuracy: 0.4987
 3872/25000 [===>..........................] - ETA: 59s - loss: 7.6943 - accuracy: 0.4982
 3904/25000 [===>..........................] - ETA: 59s - loss: 7.7020 - accuracy: 0.4977
 3936/25000 [===>..........................] - ETA: 58s - loss: 7.7095 - accuracy: 0.4972
 3968/25000 [===>..........................] - ETA: 58s - loss: 7.7130 - accuracy: 0.4970
 4000/25000 [===>..........................] - ETA: 58s - loss: 7.6973 - accuracy: 0.4980
 4032/25000 [===>..........................] - ETA: 58s - loss: 7.6932 - accuracy: 0.4983
 4064/25000 [===>..........................] - ETA: 58s - loss: 7.6779 - accuracy: 0.4993
 4096/25000 [===>..........................] - ETA: 58s - loss: 7.6966 - accuracy: 0.4980
 4128/25000 [===>..........................] - ETA: 58s - loss: 7.6963 - accuracy: 0.4981
 4160/25000 [===>..........................] - ETA: 58s - loss: 7.6998 - accuracy: 0.4978
 4192/25000 [====>.........................] - ETA: 57s - loss: 7.6922 - accuracy: 0.4983
 4224/25000 [====>.........................] - ETA: 57s - loss: 7.6775 - accuracy: 0.4993
 4256/25000 [====>.........................] - ETA: 57s - loss: 7.6810 - accuracy: 0.4991
 4288/25000 [====>.........................] - ETA: 57s - loss: 7.6845 - accuracy: 0.4988
 4320/25000 [====>.........................] - ETA: 57s - loss: 7.6986 - accuracy: 0.4979
 4352/25000 [====>.........................] - ETA: 57s - loss: 7.6842 - accuracy: 0.4989
 4384/25000 [====>.........................] - ETA: 57s - loss: 7.6841 - accuracy: 0.4989
 4416/25000 [====>.........................] - ETA: 57s - loss: 7.6875 - accuracy: 0.4986
 4448/25000 [====>.........................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4996
 4480/25000 [====>.........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 4512/25000 [====>.........................] - ETA: 56s - loss: 7.6632 - accuracy: 0.5002
 4544/25000 [====>.........................] - ETA: 56s - loss: 7.6565 - accuracy: 0.5007
 4576/25000 [====>.........................] - ETA: 56s - loss: 7.6599 - accuracy: 0.5004
 4608/25000 [====>.........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 4640/25000 [====>.........................] - ETA: 56s - loss: 7.6898 - accuracy: 0.4985
 4672/25000 [====>.........................] - ETA: 56s - loss: 7.6962 - accuracy: 0.4981
 4704/25000 [====>.........................] - ETA: 56s - loss: 7.7025 - accuracy: 0.4977
 4736/25000 [====>.........................] - ETA: 56s - loss: 7.6925 - accuracy: 0.4983
 4768/25000 [====>.........................] - ETA: 56s - loss: 7.6956 - accuracy: 0.4981
 4800/25000 [====>.........................] - ETA: 55s - loss: 7.6794 - accuracy: 0.4992
 4832/25000 [====>.........................] - ETA: 55s - loss: 7.6761 - accuracy: 0.4994
 4864/25000 [====>.........................] - ETA: 55s - loss: 7.6887 - accuracy: 0.4986
 4896/25000 [====>.........................] - ETA: 55s - loss: 7.6979 - accuracy: 0.4980
 4928/25000 [====>.........................] - ETA: 55s - loss: 7.7040 - accuracy: 0.4976
 4960/25000 [====>.........................] - ETA: 55s - loss: 7.7068 - accuracy: 0.4974
 4992/25000 [====>.........................] - ETA: 55s - loss: 7.7035 - accuracy: 0.4976
 5024/25000 [=====>........................] - ETA: 55s - loss: 7.7002 - accuracy: 0.4978
 5056/25000 [=====>........................] - ETA: 55s - loss: 7.6909 - accuracy: 0.4984
 5088/25000 [=====>........................] - ETA: 54s - loss: 7.6847 - accuracy: 0.4988
 5120/25000 [=====>........................] - ETA: 54s - loss: 7.6846 - accuracy: 0.4988
 5152/25000 [=====>........................] - ETA: 54s - loss: 7.6934 - accuracy: 0.4983
 5184/25000 [=====>........................] - ETA: 54s - loss: 7.6932 - accuracy: 0.4983
 5216/25000 [=====>........................] - ETA: 54s - loss: 7.6901 - accuracy: 0.4985
 5248/25000 [=====>........................] - ETA: 54s - loss: 7.6871 - accuracy: 0.4987
 5280/25000 [=====>........................] - ETA: 54s - loss: 7.6782 - accuracy: 0.4992
 5312/25000 [=====>........................] - ETA: 54s - loss: 7.6782 - accuracy: 0.4992
 5344/25000 [=====>........................] - ETA: 53s - loss: 7.6724 - accuracy: 0.4996
 5376/25000 [=====>........................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
 5408/25000 [=====>........................] - ETA: 53s - loss: 7.6581 - accuracy: 0.5006
 5440/25000 [=====>........................] - ETA: 53s - loss: 7.6553 - accuracy: 0.5007
 5472/25000 [=====>........................] - ETA: 53s - loss: 7.6498 - accuracy: 0.5011
 5504/25000 [=====>........................] - ETA: 53s - loss: 7.6610 - accuracy: 0.5004
 5536/25000 [=====>........................] - ETA: 53s - loss: 7.6694 - accuracy: 0.4998
 5568/25000 [=====>........................] - ETA: 53s - loss: 7.6749 - accuracy: 0.4995
 5600/25000 [=====>........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 5632/25000 [=====>........................] - ETA: 53s - loss: 7.6721 - accuracy: 0.4996
 5664/25000 [=====>........................] - ETA: 52s - loss: 7.6829 - accuracy: 0.4989
 5696/25000 [=====>........................] - ETA: 52s - loss: 7.6801 - accuracy: 0.4991
 5728/25000 [=====>........................] - ETA: 52s - loss: 7.6720 - accuracy: 0.4997
 5760/25000 [=====>........................] - ETA: 52s - loss: 7.6640 - accuracy: 0.5002
 5792/25000 [=====>........................] - ETA: 52s - loss: 7.6481 - accuracy: 0.5012
 5824/25000 [=====>........................] - ETA: 52s - loss: 7.6561 - accuracy: 0.5007
 5856/25000 [======>.......................] - ETA: 52s - loss: 7.6719 - accuracy: 0.4997
 5888/25000 [======>.......................] - ETA: 52s - loss: 7.6718 - accuracy: 0.4997
 5920/25000 [======>.......................] - ETA: 52s - loss: 7.6796 - accuracy: 0.4992
 5952/25000 [======>.......................] - ETA: 51s - loss: 7.6718 - accuracy: 0.4997
 5984/25000 [======>.......................] - ETA: 51s - loss: 7.6692 - accuracy: 0.4998
 6016/25000 [======>.......................] - ETA: 51s - loss: 7.6692 - accuracy: 0.4998
 6048/25000 [======>.......................] - ETA: 51s - loss: 7.6590 - accuracy: 0.5005
 6080/25000 [======>.......................] - ETA: 51s - loss: 7.6540 - accuracy: 0.5008
 6112/25000 [======>.......................] - ETA: 51s - loss: 7.6591 - accuracy: 0.5005
 6144/25000 [======>.......................] - ETA: 51s - loss: 7.6641 - accuracy: 0.5002
 6176/25000 [======>.......................] - ETA: 51s - loss: 7.6617 - accuracy: 0.5003
 6208/25000 [======>.......................] - ETA: 51s - loss: 7.6617 - accuracy: 0.5003
 6240/25000 [======>.......................] - ETA: 51s - loss: 7.6470 - accuracy: 0.5013
 6272/25000 [======>.......................] - ETA: 51s - loss: 7.6568 - accuracy: 0.5006
 6304/25000 [======>.......................] - ETA: 50s - loss: 7.6350 - accuracy: 0.5021
 6336/25000 [======>.......................] - ETA: 50s - loss: 7.6352 - accuracy: 0.5021
 6368/25000 [======>.......................] - ETA: 50s - loss: 7.6401 - accuracy: 0.5017
 6400/25000 [======>.......................] - ETA: 50s - loss: 7.6355 - accuracy: 0.5020
 6432/25000 [======>.......................] - ETA: 50s - loss: 7.6309 - accuracy: 0.5023
 6464/25000 [======>.......................] - ETA: 50s - loss: 7.6310 - accuracy: 0.5023
 6496/25000 [======>.......................] - ETA: 50s - loss: 7.6312 - accuracy: 0.5023
 6528/25000 [======>.......................] - ETA: 50s - loss: 7.6384 - accuracy: 0.5018
 6560/25000 [======>.......................] - ETA: 50s - loss: 7.6549 - accuracy: 0.5008
 6592/25000 [======>.......................] - ETA: 50s - loss: 7.6503 - accuracy: 0.5011
 6624/25000 [======>.......................] - ETA: 49s - loss: 7.6388 - accuracy: 0.5018
 6656/25000 [======>.......................] - ETA: 49s - loss: 7.6344 - accuracy: 0.5021
 6688/25000 [=======>......................] - ETA: 49s - loss: 7.6460 - accuracy: 0.5013
 6720/25000 [=======>......................] - ETA: 49s - loss: 7.6643 - accuracy: 0.5001
 6752/25000 [=======>......................] - ETA: 49s - loss: 7.6621 - accuracy: 0.5003
 6784/25000 [=======>......................] - ETA: 49s - loss: 7.6644 - accuracy: 0.5001
 6816/25000 [=======>......................] - ETA: 49s - loss: 7.6599 - accuracy: 0.5004
 6848/25000 [=======>......................] - ETA: 49s - loss: 7.6487 - accuracy: 0.5012
 6880/25000 [=======>......................] - ETA: 49s - loss: 7.6622 - accuracy: 0.5003
 6912/25000 [=======>......................] - ETA: 49s - loss: 7.6622 - accuracy: 0.5003
 6944/25000 [=======>......................] - ETA: 49s - loss: 7.6600 - accuracy: 0.5004
 6976/25000 [=======>......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 7008/25000 [=======>......................] - ETA: 48s - loss: 7.6601 - accuracy: 0.5004
 7040/25000 [=======>......................] - ETA: 48s - loss: 7.6557 - accuracy: 0.5007
 7072/25000 [=======>......................] - ETA: 48s - loss: 7.6601 - accuracy: 0.5004
 7104/25000 [=======>......................] - ETA: 48s - loss: 7.6688 - accuracy: 0.4999
 7136/25000 [=======>......................] - ETA: 48s - loss: 7.6731 - accuracy: 0.4996
 7168/25000 [=======>......................] - ETA: 48s - loss: 7.6688 - accuracy: 0.4999
 7200/25000 [=======>......................] - ETA: 48s - loss: 7.6687 - accuracy: 0.4999
 7232/25000 [=======>......................] - ETA: 48s - loss: 7.6645 - accuracy: 0.5001
 7264/25000 [=======>......................] - ETA: 48s - loss: 7.6645 - accuracy: 0.5001
 7296/25000 [=======>......................] - ETA: 48s - loss: 7.6645 - accuracy: 0.5001
 7328/25000 [=======>......................] - ETA: 47s - loss: 7.6645 - accuracy: 0.5001
 7360/25000 [=======>......................] - ETA: 47s - loss: 7.6583 - accuracy: 0.5005
 7392/25000 [=======>......................] - ETA: 47s - loss: 7.6728 - accuracy: 0.4996
 7424/25000 [=======>......................] - ETA: 47s - loss: 7.6708 - accuracy: 0.4997
 7456/25000 [=======>......................] - ETA: 47s - loss: 7.6790 - accuracy: 0.4992
 7488/25000 [=======>......................] - ETA: 47s - loss: 7.6830 - accuracy: 0.4989
 7520/25000 [========>.....................] - ETA: 47s - loss: 7.6829 - accuracy: 0.4989
 7552/25000 [========>.....................] - ETA: 47s - loss: 7.6747 - accuracy: 0.4995
 7584/25000 [========>.....................] - ETA: 47s - loss: 7.6788 - accuracy: 0.4992
 7616/25000 [========>.....................] - ETA: 47s - loss: 7.6747 - accuracy: 0.4995
 7648/25000 [========>.....................] - ETA: 46s - loss: 7.6807 - accuracy: 0.4991
 7680/25000 [========>.....................] - ETA: 46s - loss: 7.6706 - accuracy: 0.4997
 7712/25000 [========>.....................] - ETA: 46s - loss: 7.6766 - accuracy: 0.4994
 7744/25000 [========>.....................] - ETA: 46s - loss: 7.6726 - accuracy: 0.4996
 7776/25000 [========>.....................] - ETA: 46s - loss: 7.6765 - accuracy: 0.4994
 7808/25000 [========>.....................] - ETA: 46s - loss: 7.6764 - accuracy: 0.4994
 7840/25000 [========>.....................] - ETA: 46s - loss: 7.6764 - accuracy: 0.4994
 7872/25000 [========>.....................] - ETA: 46s - loss: 7.6783 - accuracy: 0.4992
 7904/25000 [========>.....................] - ETA: 46s - loss: 7.6802 - accuracy: 0.4991
 7936/25000 [========>.....................] - ETA: 46s - loss: 7.6840 - accuracy: 0.4989
 7968/25000 [========>.....................] - ETA: 45s - loss: 7.6762 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 45s - loss: 7.6647 - accuracy: 0.5001
 8032/25000 [========>.....................] - ETA: 45s - loss: 7.6590 - accuracy: 0.5005
 8064/25000 [========>.....................] - ETA: 45s - loss: 7.6590 - accuracy: 0.5005
 8096/25000 [========>.....................] - ETA: 45s - loss: 7.6571 - accuracy: 0.5006
 8128/25000 [========>.....................] - ETA: 45s - loss: 7.6534 - accuracy: 0.5009
 8160/25000 [========>.....................] - ETA: 45s - loss: 7.6516 - accuracy: 0.5010
 8192/25000 [========>.....................] - ETA: 45s - loss: 7.6573 - accuracy: 0.5006
 8224/25000 [========>.....................] - ETA: 45s - loss: 7.6592 - accuracy: 0.5005
 8256/25000 [========>.....................] - ETA: 45s - loss: 7.6629 - accuracy: 0.5002
 8288/25000 [========>.....................] - ETA: 44s - loss: 7.6611 - accuracy: 0.5004
 8320/25000 [========>.....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 8352/25000 [=========>....................] - ETA: 44s - loss: 7.6611 - accuracy: 0.5004
 8384/25000 [=========>....................] - ETA: 44s - loss: 7.6630 - accuracy: 0.5002
 8416/25000 [=========>....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 8448/25000 [=========>....................] - ETA: 44s - loss: 7.6684 - accuracy: 0.4999
 8480/25000 [=========>....................] - ETA: 44s - loss: 7.6702 - accuracy: 0.4998
 8512/25000 [=========>....................] - ETA: 44s - loss: 7.6720 - accuracy: 0.4996
 8544/25000 [=========>....................] - ETA: 44s - loss: 7.6684 - accuracy: 0.4999
 8576/25000 [=========>....................] - ETA: 44s - loss: 7.6630 - accuracy: 0.5002
 8608/25000 [=========>....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 8640/25000 [=========>....................] - ETA: 43s - loss: 7.6719 - accuracy: 0.4997
 8672/25000 [=========>....................] - ETA: 43s - loss: 7.6684 - accuracy: 0.4999
 8704/25000 [=========>....................] - ETA: 43s - loss: 7.6684 - accuracy: 0.4999
 8736/25000 [=========>....................] - ETA: 43s - loss: 7.6754 - accuracy: 0.4994
 8768/25000 [=========>....................] - ETA: 43s - loss: 7.6754 - accuracy: 0.4994
 8800/25000 [=========>....................] - ETA: 43s - loss: 7.6753 - accuracy: 0.4994
 8832/25000 [=========>....................] - ETA: 43s - loss: 7.6631 - accuracy: 0.5002
 8864/25000 [=========>....................] - ETA: 43s - loss: 7.6718 - accuracy: 0.4997
 8896/25000 [=========>....................] - ETA: 43s - loss: 7.6735 - accuracy: 0.4996
 8928/25000 [=========>....................] - ETA: 43s - loss: 7.6752 - accuracy: 0.4994
 8960/25000 [=========>....................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 8992/25000 [=========>....................] - ETA: 42s - loss: 7.6598 - accuracy: 0.5004
 9024/25000 [=========>....................] - ETA: 42s - loss: 7.6632 - accuracy: 0.5002
 9056/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 9088/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 9120/25000 [=========>....................] - ETA: 42s - loss: 7.6717 - accuracy: 0.4997
 9152/25000 [=========>....................] - ETA: 42s - loss: 7.6767 - accuracy: 0.4993
 9184/25000 [==========>...................] - ETA: 42s - loss: 7.6750 - accuracy: 0.4995
 9216/25000 [==========>...................] - ETA: 42s - loss: 7.6749 - accuracy: 0.4995
 9248/25000 [==========>...................] - ETA: 42s - loss: 7.6882 - accuracy: 0.4986
 9280/25000 [==========>...................] - ETA: 42s - loss: 7.6914 - accuracy: 0.4984
 9312/25000 [==========>...................] - ETA: 41s - loss: 7.6930 - accuracy: 0.4983
 9344/25000 [==========>...................] - ETA: 41s - loss: 7.6978 - accuracy: 0.4980
 9376/25000 [==========>...................] - ETA: 41s - loss: 7.6993 - accuracy: 0.4979
 9408/25000 [==========>...................] - ETA: 41s - loss: 7.7041 - accuracy: 0.4976
 9440/25000 [==========>...................] - ETA: 41s - loss: 7.7056 - accuracy: 0.4975
 9472/25000 [==========>...................] - ETA: 41s - loss: 7.7087 - accuracy: 0.4973
 9504/25000 [==========>...................] - ETA: 41s - loss: 7.7070 - accuracy: 0.4974
 9536/25000 [==========>...................] - ETA: 41s - loss: 7.7068 - accuracy: 0.4974
 9568/25000 [==========>...................] - ETA: 41s - loss: 7.7115 - accuracy: 0.4971
 9600/25000 [==========>...................] - ETA: 41s - loss: 7.7065 - accuracy: 0.4974
 9632/25000 [==========>...................] - ETA: 41s - loss: 7.7048 - accuracy: 0.4975
 9664/25000 [==========>...................] - ETA: 40s - loss: 7.7095 - accuracy: 0.4972
 9696/25000 [==========>...................] - ETA: 40s - loss: 7.6982 - accuracy: 0.4979
 9728/25000 [==========>...................] - ETA: 40s - loss: 7.6934 - accuracy: 0.4983
 9760/25000 [==========>...................] - ETA: 40s - loss: 7.6918 - accuracy: 0.4984
 9792/25000 [==========>...................] - ETA: 40s - loss: 7.6964 - accuracy: 0.4981
 9824/25000 [==========>...................] - ETA: 40s - loss: 7.7010 - accuracy: 0.4978
 9856/25000 [==========>...................] - ETA: 40s - loss: 7.7055 - accuracy: 0.4975
 9888/25000 [==========>...................] - ETA: 40s - loss: 7.7023 - accuracy: 0.4977
 9920/25000 [==========>...................] - ETA: 40s - loss: 7.7068 - accuracy: 0.4974
 9952/25000 [==========>...................] - ETA: 40s - loss: 7.7113 - accuracy: 0.4971
 9984/25000 [==========>...................] - ETA: 40s - loss: 7.7127 - accuracy: 0.4970
10016/25000 [===========>..................] - ETA: 39s - loss: 7.7034 - accuracy: 0.4976
10048/25000 [===========>..................] - ETA: 39s - loss: 7.7063 - accuracy: 0.4974
10080/25000 [===========>..................] - ETA: 39s - loss: 7.7107 - accuracy: 0.4971
10112/25000 [===========>..................] - ETA: 39s - loss: 7.7182 - accuracy: 0.4966
10144/25000 [===========>..................] - ETA: 39s - loss: 7.7210 - accuracy: 0.4965
10176/25000 [===========>..................] - ETA: 39s - loss: 7.7254 - accuracy: 0.4962
10208/25000 [===========>..................] - ETA: 39s - loss: 7.7237 - accuracy: 0.4963
10240/25000 [===========>..................] - ETA: 39s - loss: 7.7280 - accuracy: 0.4960
10272/25000 [===========>..................] - ETA: 39s - loss: 7.7338 - accuracy: 0.4956
10304/25000 [===========>..................] - ETA: 39s - loss: 7.7380 - accuracy: 0.4953
10336/25000 [===========>..................] - ETA: 39s - loss: 7.7438 - accuracy: 0.4950
10368/25000 [===========>..................] - ETA: 38s - loss: 7.7406 - accuracy: 0.4952
10400/25000 [===========>..................] - ETA: 38s - loss: 7.7359 - accuracy: 0.4955
10432/25000 [===========>..................] - ETA: 38s - loss: 7.7372 - accuracy: 0.4954
10464/25000 [===========>..................] - ETA: 38s - loss: 7.7428 - accuracy: 0.4950
10496/25000 [===========>..................] - ETA: 38s - loss: 7.7411 - accuracy: 0.4951
10528/25000 [===========>..................] - ETA: 38s - loss: 7.7351 - accuracy: 0.4955
10560/25000 [===========>..................] - ETA: 38s - loss: 7.7363 - accuracy: 0.4955
10592/25000 [===========>..................] - ETA: 38s - loss: 7.7318 - accuracy: 0.4958
10624/25000 [===========>..................] - ETA: 38s - loss: 7.7301 - accuracy: 0.4959
10656/25000 [===========>..................] - ETA: 38s - loss: 7.7299 - accuracy: 0.4959
10688/25000 [===========>..................] - ETA: 38s - loss: 7.7312 - accuracy: 0.4958
10720/25000 [===========>..................] - ETA: 37s - loss: 7.7296 - accuracy: 0.4959
10752/25000 [===========>..................] - ETA: 37s - loss: 7.7294 - accuracy: 0.4959
10784/25000 [===========>..................] - ETA: 37s - loss: 7.7263 - accuracy: 0.4961
10816/25000 [===========>..................] - ETA: 37s - loss: 7.7262 - accuracy: 0.4961
10848/25000 [============>.................] - ETA: 37s - loss: 7.7274 - accuracy: 0.4960
10880/25000 [============>.................] - ETA: 37s - loss: 7.7258 - accuracy: 0.4961
10912/25000 [============>.................] - ETA: 37s - loss: 7.7228 - accuracy: 0.4963
10944/25000 [============>.................] - ETA: 37s - loss: 7.7199 - accuracy: 0.4965
10976/25000 [============>.................] - ETA: 37s - loss: 7.7253 - accuracy: 0.4962
11008/25000 [============>.................] - ETA: 37s - loss: 7.7265 - accuracy: 0.4961
11040/25000 [============>.................] - ETA: 37s - loss: 7.7180 - accuracy: 0.4966
11072/25000 [============>.................] - ETA: 36s - loss: 7.7151 - accuracy: 0.4968
11104/25000 [============>.................] - ETA: 36s - loss: 7.7191 - accuracy: 0.4966
11136/25000 [============>.................] - ETA: 36s - loss: 7.7231 - accuracy: 0.4963
11168/25000 [============>.................] - ETA: 36s - loss: 7.7202 - accuracy: 0.4965
11200/25000 [============>.................] - ETA: 36s - loss: 7.7186 - accuracy: 0.4966
11232/25000 [============>.................] - ETA: 36s - loss: 7.7130 - accuracy: 0.4970
11264/25000 [============>.................] - ETA: 36s - loss: 7.7143 - accuracy: 0.4969
11296/25000 [============>.................] - ETA: 36s - loss: 7.7182 - accuracy: 0.4966
11328/25000 [============>.................] - ETA: 36s - loss: 7.7167 - accuracy: 0.4967
11360/25000 [============>.................] - ETA: 36s - loss: 7.7152 - accuracy: 0.4968
11392/25000 [============>.................] - ETA: 36s - loss: 7.7232 - accuracy: 0.4963
11424/25000 [============>.................] - ETA: 35s - loss: 7.7243 - accuracy: 0.4962
11456/25000 [============>.................] - ETA: 35s - loss: 7.7268 - accuracy: 0.4961
11488/25000 [============>.................] - ETA: 35s - loss: 7.7240 - accuracy: 0.4963
11520/25000 [============>.................] - ETA: 35s - loss: 7.7212 - accuracy: 0.4964
11552/25000 [============>.................] - ETA: 35s - loss: 7.7157 - accuracy: 0.4968
11584/25000 [============>.................] - ETA: 35s - loss: 7.7169 - accuracy: 0.4967
11616/25000 [============>.................] - ETA: 35s - loss: 7.7207 - accuracy: 0.4965
11648/25000 [============>.................] - ETA: 35s - loss: 7.7180 - accuracy: 0.4967
11680/25000 [=============>................] - ETA: 35s - loss: 7.7218 - accuracy: 0.4964
11712/25000 [=============>................] - ETA: 35s - loss: 7.7216 - accuracy: 0.4964
11744/25000 [=============>................] - ETA: 35s - loss: 7.7241 - accuracy: 0.4963
11776/25000 [=============>................] - ETA: 34s - loss: 7.7252 - accuracy: 0.4962
11808/25000 [=============>................] - ETA: 34s - loss: 7.7251 - accuracy: 0.4962
11840/25000 [=============>................] - ETA: 34s - loss: 7.7249 - accuracy: 0.4962
11872/25000 [=============>................] - ETA: 34s - loss: 7.7209 - accuracy: 0.4965
11904/25000 [=============>................] - ETA: 34s - loss: 7.7259 - accuracy: 0.4961
11936/25000 [=============>................] - ETA: 34s - loss: 7.7219 - accuracy: 0.4964
11968/25000 [=============>................] - ETA: 34s - loss: 7.7191 - accuracy: 0.4966
12000/25000 [=============>................] - ETA: 34s - loss: 7.7190 - accuracy: 0.4966
12032/25000 [=============>................] - ETA: 34s - loss: 7.7112 - accuracy: 0.4971
12064/25000 [=============>................] - ETA: 34s - loss: 7.7111 - accuracy: 0.4971
12096/25000 [=============>................] - ETA: 34s - loss: 7.7034 - accuracy: 0.4976
12128/25000 [=============>................] - ETA: 33s - loss: 7.7058 - accuracy: 0.4974
12160/25000 [=============>................] - ETA: 33s - loss: 7.7032 - accuracy: 0.4976
12192/25000 [=============>................] - ETA: 33s - loss: 7.7031 - accuracy: 0.4976
12224/25000 [=============>................] - ETA: 33s - loss: 7.7005 - accuracy: 0.4978
12256/25000 [=============>................] - ETA: 33s - loss: 7.7054 - accuracy: 0.4975
12288/25000 [=============>................] - ETA: 33s - loss: 7.7090 - accuracy: 0.4972
12320/25000 [=============>................] - ETA: 33s - loss: 7.7164 - accuracy: 0.4968
12352/25000 [=============>................] - ETA: 33s - loss: 7.7163 - accuracy: 0.4968
12384/25000 [=============>................] - ETA: 33s - loss: 7.7186 - accuracy: 0.4966
12416/25000 [=============>................] - ETA: 33s - loss: 7.7160 - accuracy: 0.4968
12448/25000 [=============>................] - ETA: 33s - loss: 7.7159 - accuracy: 0.4968
12480/25000 [=============>................] - ETA: 33s - loss: 7.7133 - accuracy: 0.4970
12512/25000 [==============>...............] - ETA: 32s - loss: 7.7193 - accuracy: 0.4966
12544/25000 [==============>...............] - ETA: 32s - loss: 7.7216 - accuracy: 0.4964
12576/25000 [==============>...............] - ETA: 32s - loss: 7.7276 - accuracy: 0.4960
12608/25000 [==============>...............] - ETA: 32s - loss: 7.7299 - accuracy: 0.4959
12640/25000 [==============>...............] - ETA: 32s - loss: 7.7333 - accuracy: 0.4956
12672/25000 [==============>...............] - ETA: 32s - loss: 7.7368 - accuracy: 0.4954
12704/25000 [==============>...............] - ETA: 32s - loss: 7.7390 - accuracy: 0.4953
12736/25000 [==============>...............] - ETA: 32s - loss: 7.7401 - accuracy: 0.4952
12768/25000 [==============>...............] - ETA: 32s - loss: 7.7399 - accuracy: 0.4952
12800/25000 [==============>...............] - ETA: 32s - loss: 7.7445 - accuracy: 0.4949
12832/25000 [==============>...............] - ETA: 32s - loss: 7.7491 - accuracy: 0.4946
12864/25000 [==============>...............] - ETA: 31s - loss: 7.7501 - accuracy: 0.4946
12896/25000 [==============>...............] - ETA: 31s - loss: 7.7546 - accuracy: 0.4943
12928/25000 [==============>...............] - ETA: 31s - loss: 7.7544 - accuracy: 0.4943
12960/25000 [==============>...............] - ETA: 31s - loss: 7.7518 - accuracy: 0.4944
12992/25000 [==============>...............] - ETA: 31s - loss: 7.7540 - accuracy: 0.4943
13024/25000 [==============>...............] - ETA: 31s - loss: 7.7537 - accuracy: 0.4943
13056/25000 [==============>...............] - ETA: 31s - loss: 7.7582 - accuracy: 0.4940
13088/25000 [==============>...............] - ETA: 31s - loss: 7.7603 - accuracy: 0.4939
13120/25000 [==============>...............] - ETA: 31s - loss: 7.7543 - accuracy: 0.4943
13152/25000 [==============>...............] - ETA: 31s - loss: 7.7541 - accuracy: 0.4943
13184/25000 [==============>...............] - ETA: 31s - loss: 7.7527 - accuracy: 0.4944
13216/25000 [==============>...............] - ETA: 31s - loss: 7.7478 - accuracy: 0.4947
13248/25000 [==============>...............] - ETA: 30s - loss: 7.7476 - accuracy: 0.4947
13280/25000 [==============>...............] - ETA: 30s - loss: 7.7440 - accuracy: 0.4950
13312/25000 [==============>...............] - ETA: 30s - loss: 7.7426 - accuracy: 0.4950
13344/25000 [===============>..............] - ETA: 30s - loss: 7.7413 - accuracy: 0.4951
13376/25000 [===============>..............] - ETA: 30s - loss: 7.7343 - accuracy: 0.4956
13408/25000 [===============>..............] - ETA: 30s - loss: 7.7329 - accuracy: 0.4957
13440/25000 [===============>..............] - ETA: 30s - loss: 7.7351 - accuracy: 0.4955
13472/25000 [===============>..............] - ETA: 30s - loss: 7.7360 - accuracy: 0.4955
13504/25000 [===============>..............] - ETA: 30s - loss: 7.7370 - accuracy: 0.4954
13536/25000 [===============>..............] - ETA: 30s - loss: 7.7346 - accuracy: 0.4956
13568/25000 [===============>..............] - ETA: 30s - loss: 7.7322 - accuracy: 0.4957
13600/25000 [===============>..............] - ETA: 29s - loss: 7.7286 - accuracy: 0.4960
13632/25000 [===============>..............] - ETA: 29s - loss: 7.7251 - accuracy: 0.4962
13664/25000 [===============>..............] - ETA: 29s - loss: 7.7205 - accuracy: 0.4965
13696/25000 [===============>..............] - ETA: 29s - loss: 7.7192 - accuracy: 0.4966
13728/25000 [===============>..............] - ETA: 29s - loss: 7.7180 - accuracy: 0.4966
13760/25000 [===============>..............] - ETA: 29s - loss: 7.7123 - accuracy: 0.4970
13792/25000 [===============>..............] - ETA: 29s - loss: 7.7133 - accuracy: 0.4970
13824/25000 [===============>..............] - ETA: 29s - loss: 7.7199 - accuracy: 0.4965
13856/25000 [===============>..............] - ETA: 29s - loss: 7.7231 - accuracy: 0.4963
13888/25000 [===============>..............] - ETA: 29s - loss: 7.7185 - accuracy: 0.4966
13920/25000 [===============>..............] - ETA: 29s - loss: 7.7217 - accuracy: 0.4964
13952/25000 [===============>..............] - ETA: 29s - loss: 7.7249 - accuracy: 0.4962
13984/25000 [===============>..............] - ETA: 28s - loss: 7.7302 - accuracy: 0.4959
14016/25000 [===============>..............] - ETA: 28s - loss: 7.7355 - accuracy: 0.4955
14048/25000 [===============>..............] - ETA: 28s - loss: 7.7343 - accuracy: 0.4956
14080/25000 [===============>..............] - ETA: 28s - loss: 7.7341 - accuracy: 0.4956
14112/25000 [===============>..............] - ETA: 28s - loss: 7.7372 - accuracy: 0.4954
14144/25000 [===============>..............] - ETA: 28s - loss: 7.7414 - accuracy: 0.4951
14176/25000 [================>.............] - ETA: 28s - loss: 7.7423 - accuracy: 0.4951
14208/25000 [================>.............] - ETA: 28s - loss: 7.7346 - accuracy: 0.4956
14240/25000 [================>.............] - ETA: 28s - loss: 7.7377 - accuracy: 0.4954
14272/25000 [================>.............] - ETA: 28s - loss: 7.7343 - accuracy: 0.4956
14304/25000 [================>.............] - ETA: 28s - loss: 7.7363 - accuracy: 0.4955
14336/25000 [================>.............] - ETA: 28s - loss: 7.7361 - accuracy: 0.4955
14368/25000 [================>.............] - ETA: 27s - loss: 7.7413 - accuracy: 0.4951
14400/25000 [================>.............] - ETA: 27s - loss: 7.7422 - accuracy: 0.4951
14432/25000 [================>.............] - ETA: 27s - loss: 7.7442 - accuracy: 0.4949
14464/25000 [================>.............] - ETA: 27s - loss: 7.7355 - accuracy: 0.4955
14496/25000 [================>.............] - ETA: 27s - loss: 7.7311 - accuracy: 0.4958
14528/25000 [================>.............] - ETA: 27s - loss: 7.7373 - accuracy: 0.4954
14560/25000 [================>.............] - ETA: 27s - loss: 7.7393 - accuracy: 0.4953
14592/25000 [================>.............] - ETA: 27s - loss: 7.7412 - accuracy: 0.4951
14624/25000 [================>.............] - ETA: 27s - loss: 7.7400 - accuracy: 0.4952
14656/25000 [================>.............] - ETA: 27s - loss: 7.7409 - accuracy: 0.4952
14688/25000 [================>.............] - ETA: 27s - loss: 7.7334 - accuracy: 0.4956
14720/25000 [================>.............] - ETA: 26s - loss: 7.7312 - accuracy: 0.4958
14752/25000 [================>.............] - ETA: 26s - loss: 7.7321 - accuracy: 0.4957
14784/25000 [================>.............] - ETA: 26s - loss: 7.7351 - accuracy: 0.4955
14816/25000 [================>.............] - ETA: 26s - loss: 7.7411 - accuracy: 0.4951
14848/25000 [================>.............] - ETA: 26s - loss: 7.7379 - accuracy: 0.4954
14880/25000 [================>.............] - ETA: 26s - loss: 7.7336 - accuracy: 0.4956
14912/25000 [================>.............] - ETA: 26s - loss: 7.7345 - accuracy: 0.4956
14944/25000 [================>.............] - ETA: 26s - loss: 7.7323 - accuracy: 0.4957
14976/25000 [================>.............] - ETA: 26s - loss: 7.7332 - accuracy: 0.4957
15008/25000 [=================>............] - ETA: 26s - loss: 7.7310 - accuracy: 0.4958
15040/25000 [=================>............] - ETA: 26s - loss: 7.7247 - accuracy: 0.4962
15072/25000 [=================>............] - ETA: 26s - loss: 7.7277 - accuracy: 0.4960
15104/25000 [=================>............] - ETA: 25s - loss: 7.7275 - accuracy: 0.4960
15136/25000 [=================>............] - ETA: 25s - loss: 7.7294 - accuracy: 0.4959
15168/25000 [=================>............] - ETA: 25s - loss: 7.7283 - accuracy: 0.4960
15200/25000 [=================>............] - ETA: 25s - loss: 7.7282 - accuracy: 0.4960
15232/25000 [=================>............] - ETA: 25s - loss: 7.7280 - accuracy: 0.4960
15264/25000 [=================>............] - ETA: 25s - loss: 7.7299 - accuracy: 0.4959
15296/25000 [=================>............] - ETA: 25s - loss: 7.7358 - accuracy: 0.4955
15328/25000 [=================>............] - ETA: 25s - loss: 7.7406 - accuracy: 0.4952
15360/25000 [=================>............] - ETA: 25s - loss: 7.7335 - accuracy: 0.4956
15392/25000 [=================>............] - ETA: 25s - loss: 7.7334 - accuracy: 0.4956
15424/25000 [=================>............] - ETA: 25s - loss: 7.7332 - accuracy: 0.4957
15456/25000 [=================>............] - ETA: 25s - loss: 7.7361 - accuracy: 0.4955
15488/25000 [=================>............] - ETA: 24s - loss: 7.7369 - accuracy: 0.4954
15520/25000 [=================>............] - ETA: 24s - loss: 7.7348 - accuracy: 0.4956
15552/25000 [=================>............] - ETA: 24s - loss: 7.7327 - accuracy: 0.4957
15584/25000 [=================>............] - ETA: 24s - loss: 7.7365 - accuracy: 0.4954
15616/25000 [=================>............] - ETA: 24s - loss: 7.7383 - accuracy: 0.4953
15648/25000 [=================>............] - ETA: 24s - loss: 7.7382 - accuracy: 0.4953
15680/25000 [=================>............] - ETA: 24s - loss: 7.7390 - accuracy: 0.4953
15712/25000 [=================>............] - ETA: 24s - loss: 7.7369 - accuracy: 0.4954
15744/25000 [=================>............] - ETA: 24s - loss: 7.7367 - accuracy: 0.4954
15776/25000 [=================>............] - ETA: 24s - loss: 7.7395 - accuracy: 0.4952
15808/25000 [=================>............] - ETA: 24s - loss: 7.7403 - accuracy: 0.4952
15840/25000 [==================>...........] - ETA: 24s - loss: 7.7412 - accuracy: 0.4951
15872/25000 [==================>...........] - ETA: 23s - loss: 7.7362 - accuracy: 0.4955
15904/25000 [==================>...........] - ETA: 23s - loss: 7.7341 - accuracy: 0.4956
15936/25000 [==================>...........] - ETA: 23s - loss: 7.7378 - accuracy: 0.4954
15968/25000 [==================>...........] - ETA: 23s - loss: 7.7358 - accuracy: 0.4955
16000/25000 [==================>...........] - ETA: 23s - loss: 7.7356 - accuracy: 0.4955
16032/25000 [==================>...........] - ETA: 23s - loss: 7.7336 - accuracy: 0.4956
16064/25000 [==================>...........] - ETA: 23s - loss: 7.7334 - accuracy: 0.4956
16096/25000 [==================>...........] - ETA: 23s - loss: 7.7333 - accuracy: 0.4957
16128/25000 [==================>...........] - ETA: 23s - loss: 7.7313 - accuracy: 0.4958
16160/25000 [==================>...........] - ETA: 23s - loss: 7.7349 - accuracy: 0.4955
16192/25000 [==================>...........] - ETA: 23s - loss: 7.7357 - accuracy: 0.4955
16224/25000 [==================>...........] - ETA: 22s - loss: 7.7318 - accuracy: 0.4957
16256/25000 [==================>...........] - ETA: 22s - loss: 7.7279 - accuracy: 0.4960
16288/25000 [==================>...........] - ETA: 22s - loss: 7.7288 - accuracy: 0.4959
16320/25000 [==================>...........] - ETA: 22s - loss: 7.7296 - accuracy: 0.4959
16352/25000 [==================>...........] - ETA: 22s - loss: 7.7266 - accuracy: 0.4961
16384/25000 [==================>...........] - ETA: 22s - loss: 7.7256 - accuracy: 0.4962
16416/25000 [==================>...........] - ETA: 22s - loss: 7.7217 - accuracy: 0.4964
16448/25000 [==================>...........] - ETA: 22s - loss: 7.7170 - accuracy: 0.4967
16480/25000 [==================>...........] - ETA: 22s - loss: 7.7197 - accuracy: 0.4965
16512/25000 [==================>...........] - ETA: 22s - loss: 7.7205 - accuracy: 0.4965
16544/25000 [==================>...........] - ETA: 22s - loss: 7.7213 - accuracy: 0.4964
16576/25000 [==================>...........] - ETA: 22s - loss: 7.7230 - accuracy: 0.4963
16608/25000 [==================>...........] - ETA: 21s - loss: 7.7229 - accuracy: 0.4963
16640/25000 [==================>...........] - ETA: 21s - loss: 7.7256 - accuracy: 0.4962
16672/25000 [===================>..........] - ETA: 21s - loss: 7.7236 - accuracy: 0.4963
16704/25000 [===================>..........] - ETA: 21s - loss: 7.7217 - accuracy: 0.4964
16736/25000 [===================>..........] - ETA: 21s - loss: 7.7216 - accuracy: 0.4964
16768/25000 [===================>..........] - ETA: 21s - loss: 7.7242 - accuracy: 0.4962
16800/25000 [===================>..........] - ETA: 21s - loss: 7.7259 - accuracy: 0.4961
16832/25000 [===================>..........] - ETA: 21s - loss: 7.7222 - accuracy: 0.4964
16864/25000 [===================>..........] - ETA: 21s - loss: 7.7194 - accuracy: 0.4966
16896/25000 [===================>..........] - ETA: 21s - loss: 7.7229 - accuracy: 0.4963
16928/25000 [===================>..........] - ETA: 21s - loss: 7.7264 - accuracy: 0.4961
16960/25000 [===================>..........] - ETA: 21s - loss: 7.7218 - accuracy: 0.4964
16992/25000 [===================>..........] - ETA: 20s - loss: 7.7208 - accuracy: 0.4965
17024/25000 [===================>..........] - ETA: 20s - loss: 7.7243 - accuracy: 0.4962
17056/25000 [===================>..........] - ETA: 20s - loss: 7.7215 - accuracy: 0.4964
17088/25000 [===================>..........] - ETA: 20s - loss: 7.7205 - accuracy: 0.4965
17120/25000 [===================>..........] - ETA: 20s - loss: 7.7195 - accuracy: 0.4966
17152/25000 [===================>..........] - ETA: 20s - loss: 7.7158 - accuracy: 0.4968
17184/25000 [===================>..........] - ETA: 20s - loss: 7.7175 - accuracy: 0.4967
17216/25000 [===================>..........] - ETA: 20s - loss: 7.7183 - accuracy: 0.4966
17248/25000 [===================>..........] - ETA: 20s - loss: 7.7200 - accuracy: 0.4965
17280/25000 [===================>..........] - ETA: 20s - loss: 7.7234 - accuracy: 0.4963
17312/25000 [===================>..........] - ETA: 20s - loss: 7.7268 - accuracy: 0.4961
17344/25000 [===================>..........] - ETA: 20s - loss: 7.7294 - accuracy: 0.4959
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7328 - accuracy: 0.4957
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7327 - accuracy: 0.4957
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7334 - accuracy: 0.4956
17472/25000 [===================>..........] - ETA: 19s - loss: 7.7316 - accuracy: 0.4958
17504/25000 [====================>.........] - ETA: 19s - loss: 7.7314 - accuracy: 0.4958
17536/25000 [====================>.........] - ETA: 19s - loss: 7.7348 - accuracy: 0.4956
17568/25000 [====================>.........] - ETA: 19s - loss: 7.7303 - accuracy: 0.4958
17600/25000 [====================>.........] - ETA: 19s - loss: 7.7302 - accuracy: 0.4959
17632/25000 [====================>.........] - ETA: 19s - loss: 7.7327 - accuracy: 0.4957
17664/25000 [====================>.........] - ETA: 19s - loss: 7.7300 - accuracy: 0.4959
17696/25000 [====================>.........] - ETA: 19s - loss: 7.7299 - accuracy: 0.4959
17728/25000 [====================>.........] - ETA: 18s - loss: 7.7332 - accuracy: 0.4957
17760/25000 [====================>.........] - ETA: 18s - loss: 7.7305 - accuracy: 0.4958
17792/25000 [====================>.........] - ETA: 18s - loss: 7.7313 - accuracy: 0.4958
17824/25000 [====================>.........] - ETA: 18s - loss: 7.7303 - accuracy: 0.4958
17856/25000 [====================>.........] - ETA: 18s - loss: 7.7302 - accuracy: 0.4959
17888/25000 [====================>.........] - ETA: 18s - loss: 7.7335 - accuracy: 0.4956
17920/25000 [====================>.........] - ETA: 18s - loss: 7.7325 - accuracy: 0.4957
17952/25000 [====================>.........] - ETA: 18s - loss: 7.7307 - accuracy: 0.4958
17984/25000 [====================>.........] - ETA: 18s - loss: 7.7331 - accuracy: 0.4957
18016/25000 [====================>.........] - ETA: 18s - loss: 7.7330 - accuracy: 0.4957
18048/25000 [====================>.........] - ETA: 18s - loss: 7.7286 - accuracy: 0.4960
18080/25000 [====================>.........] - ETA: 18s - loss: 7.7370 - accuracy: 0.4954
18112/25000 [====================>.........] - ETA: 17s - loss: 7.7394 - accuracy: 0.4953
18144/25000 [====================>.........] - ETA: 17s - loss: 7.7385 - accuracy: 0.4953
18176/25000 [====================>.........] - ETA: 17s - loss: 7.7417 - accuracy: 0.4951
18208/25000 [====================>.........] - ETA: 17s - loss: 7.7407 - accuracy: 0.4952
18240/25000 [====================>.........] - ETA: 17s - loss: 7.7431 - accuracy: 0.4950
18272/25000 [====================>.........] - ETA: 17s - loss: 7.7430 - accuracy: 0.4950
18304/25000 [====================>.........] - ETA: 17s - loss: 7.7412 - accuracy: 0.4951
18336/25000 [=====================>........] - ETA: 17s - loss: 7.7385 - accuracy: 0.4953
18368/25000 [=====================>........] - ETA: 17s - loss: 7.7384 - accuracy: 0.4953
18400/25000 [=====================>........] - ETA: 17s - loss: 7.7366 - accuracy: 0.4954
18432/25000 [=====================>........] - ETA: 17s - loss: 7.7382 - accuracy: 0.4953
18464/25000 [=====================>........] - ETA: 17s - loss: 7.7380 - accuracy: 0.4953
18496/25000 [=====================>........] - ETA: 16s - loss: 7.7363 - accuracy: 0.4955
18528/25000 [=====================>........] - ETA: 16s - loss: 7.7337 - accuracy: 0.4956
18560/25000 [=====================>........] - ETA: 16s - loss: 7.7327 - accuracy: 0.4957
18592/25000 [=====================>........] - ETA: 16s - loss: 7.7318 - accuracy: 0.4958
18624/25000 [=====================>........] - ETA: 16s - loss: 7.7366 - accuracy: 0.4954
18656/25000 [=====================>........] - ETA: 16s - loss: 7.7348 - accuracy: 0.4956
18688/25000 [=====================>........] - ETA: 16s - loss: 7.7339 - accuracy: 0.4956
18720/25000 [=====================>........] - ETA: 16s - loss: 7.7354 - accuracy: 0.4955
18752/25000 [=====================>........] - ETA: 16s - loss: 7.7329 - accuracy: 0.4957
18784/25000 [=====================>........] - ETA: 16s - loss: 7.7336 - accuracy: 0.4956
18816/25000 [=====================>........] - ETA: 16s - loss: 7.7302 - accuracy: 0.4959
18848/25000 [=====================>........] - ETA: 16s - loss: 7.7293 - accuracy: 0.4959
18880/25000 [=====================>........] - ETA: 15s - loss: 7.7292 - accuracy: 0.4959
18912/25000 [=====================>........] - ETA: 15s - loss: 7.7266 - accuracy: 0.4961
18944/25000 [=====================>........] - ETA: 15s - loss: 7.7265 - accuracy: 0.4961
18976/25000 [=====================>........] - ETA: 15s - loss: 7.7256 - accuracy: 0.4962
19008/25000 [=====================>........] - ETA: 15s - loss: 7.7231 - accuracy: 0.4963
19040/25000 [=====================>........] - ETA: 15s - loss: 7.7214 - accuracy: 0.4964
19072/25000 [=====================>........] - ETA: 15s - loss: 7.7229 - accuracy: 0.4963
19104/25000 [=====================>........] - ETA: 15s - loss: 7.7204 - accuracy: 0.4965
19136/25000 [=====================>........] - ETA: 15s - loss: 7.7203 - accuracy: 0.4965
19168/25000 [======================>.......] - ETA: 15s - loss: 7.7210 - accuracy: 0.4965
19200/25000 [======================>.......] - ETA: 15s - loss: 7.7217 - accuracy: 0.4964
19232/25000 [======================>.......] - ETA: 15s - loss: 7.7216 - accuracy: 0.4964
19264/25000 [======================>.......] - ETA: 14s - loss: 7.7215 - accuracy: 0.4964
19296/25000 [======================>.......] - ETA: 14s - loss: 7.7230 - accuracy: 0.4963
19328/25000 [======================>.......] - ETA: 14s - loss: 7.7237 - accuracy: 0.4963
19360/25000 [======================>.......] - ETA: 14s - loss: 7.7260 - accuracy: 0.4961
19392/25000 [======================>.......] - ETA: 14s - loss: 7.7235 - accuracy: 0.4963
19424/25000 [======================>.......] - ETA: 14s - loss: 7.7242 - accuracy: 0.4962
19456/25000 [======================>.......] - ETA: 14s - loss: 7.7218 - accuracy: 0.4964
19488/25000 [======================>.......] - ETA: 14s - loss: 7.7233 - accuracy: 0.4963
19520/25000 [======================>.......] - ETA: 14s - loss: 7.7271 - accuracy: 0.4961
19552/25000 [======================>.......] - ETA: 14s - loss: 7.7294 - accuracy: 0.4959
19584/25000 [======================>.......] - ETA: 14s - loss: 7.7285 - accuracy: 0.4960
19616/25000 [======================>.......] - ETA: 14s - loss: 7.7276 - accuracy: 0.4960
19648/25000 [======================>.......] - ETA: 13s - loss: 7.7236 - accuracy: 0.4963
19680/25000 [======================>.......] - ETA: 13s - loss: 7.7235 - accuracy: 0.4963
19712/25000 [======================>.......] - ETA: 13s - loss: 7.7226 - accuracy: 0.4963
19744/25000 [======================>.......] - ETA: 13s - loss: 7.7249 - accuracy: 0.4962
19776/25000 [======================>.......] - ETA: 13s - loss: 7.7240 - accuracy: 0.4963
19808/25000 [======================>.......] - ETA: 13s - loss: 7.7216 - accuracy: 0.4964
19840/25000 [======================>.......] - ETA: 13s - loss: 7.7207 - accuracy: 0.4965
19872/25000 [======================>.......] - ETA: 13s - loss: 7.7191 - accuracy: 0.4966
19904/25000 [======================>.......] - ETA: 13s - loss: 7.7175 - accuracy: 0.4967
19936/25000 [======================>.......] - ETA: 13s - loss: 7.7166 - accuracy: 0.4967
19968/25000 [======================>.......] - ETA: 13s - loss: 7.7181 - accuracy: 0.4966
20000/25000 [=======================>......] - ETA: 13s - loss: 7.7149 - accuracy: 0.4969
20032/25000 [=======================>......] - ETA: 12s - loss: 7.7141 - accuracy: 0.4969
20064/25000 [=======================>......] - ETA: 12s - loss: 7.7148 - accuracy: 0.4969
20096/25000 [=======================>......] - ETA: 12s - loss: 7.7116 - accuracy: 0.4971
20128/25000 [=======================>......] - ETA: 12s - loss: 7.7131 - accuracy: 0.4970
20160/25000 [=======================>......] - ETA: 12s - loss: 7.7153 - accuracy: 0.4968
20192/25000 [=======================>......] - ETA: 12s - loss: 7.7114 - accuracy: 0.4971
20224/25000 [=======================>......] - ETA: 12s - loss: 7.7121 - accuracy: 0.4970
20256/25000 [=======================>......] - ETA: 12s - loss: 7.7143 - accuracy: 0.4969
20288/25000 [=======================>......] - ETA: 12s - loss: 7.7127 - accuracy: 0.4970
20320/25000 [=======================>......] - ETA: 12s - loss: 7.7111 - accuracy: 0.4971
20352/25000 [=======================>......] - ETA: 12s - loss: 7.7126 - accuracy: 0.4970
20384/25000 [=======================>......] - ETA: 11s - loss: 7.7155 - accuracy: 0.4968
20416/25000 [=======================>......] - ETA: 11s - loss: 7.7117 - accuracy: 0.4971
20448/25000 [=======================>......] - ETA: 11s - loss: 7.7124 - accuracy: 0.4970
20480/25000 [=======================>......] - ETA: 11s - loss: 7.7123 - accuracy: 0.4970
20512/25000 [=======================>......] - ETA: 11s - loss: 7.7137 - accuracy: 0.4969
20544/25000 [=======================>......] - ETA: 11s - loss: 7.7129 - accuracy: 0.4970
20576/25000 [=======================>......] - ETA: 11s - loss: 7.7136 - accuracy: 0.4969
20608/25000 [=======================>......] - ETA: 11s - loss: 7.7120 - accuracy: 0.4970
20640/25000 [=======================>......] - ETA: 11s - loss: 7.7104 - accuracy: 0.4971
20672/25000 [=======================>......] - ETA: 11s - loss: 7.7074 - accuracy: 0.4973
20704/25000 [=======================>......] - ETA: 11s - loss: 7.7103 - accuracy: 0.4972
20736/25000 [=======================>......] - ETA: 11s - loss: 7.7132 - accuracy: 0.4970
20768/25000 [=======================>......] - ETA: 10s - loss: 7.7139 - accuracy: 0.4969
20800/25000 [=======================>......] - ETA: 10s - loss: 7.7116 - accuracy: 0.4971
20832/25000 [=======================>......] - ETA: 10s - loss: 7.7093 - accuracy: 0.4972
20864/25000 [========================>.....] - ETA: 10s - loss: 7.7078 - accuracy: 0.4973
20896/25000 [========================>.....] - ETA: 10s - loss: 7.7062 - accuracy: 0.4974
20928/25000 [========================>.....] - ETA: 10s - loss: 7.7040 - accuracy: 0.4976
20960/25000 [========================>.....] - ETA: 10s - loss: 7.7003 - accuracy: 0.4978
20992/25000 [========================>.....] - ETA: 10s - loss: 7.7009 - accuracy: 0.4978
21024/25000 [========================>.....] - ETA: 10s - loss: 7.7009 - accuracy: 0.4978
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6994 - accuracy: 0.4979
21088/25000 [========================>.....] - ETA: 10s - loss: 7.7030 - accuracy: 0.4976
21120/25000 [========================>.....] - ETA: 10s - loss: 7.7022 - accuracy: 0.4977
21152/25000 [========================>.....] - ETA: 9s - loss: 7.7000 - accuracy: 0.4978 
21184/25000 [========================>.....] - ETA: 9s - loss: 7.7006 - accuracy: 0.4978
21216/25000 [========================>.....] - ETA: 9s - loss: 7.7020 - accuracy: 0.4977
21248/25000 [========================>.....] - ETA: 9s - loss: 7.7034 - accuracy: 0.4976
21280/25000 [========================>.....] - ETA: 9s - loss: 7.7077 - accuracy: 0.4973
21312/25000 [========================>.....] - ETA: 9s - loss: 7.7076 - accuracy: 0.4973
21344/25000 [========================>.....] - ETA: 9s - loss: 7.7076 - accuracy: 0.4973
21376/25000 [========================>.....] - ETA: 9s - loss: 7.7075 - accuracy: 0.4973
21408/25000 [========================>.....] - ETA: 9s - loss: 7.7074 - accuracy: 0.4973
21440/25000 [========================>.....] - ETA: 9s - loss: 7.7074 - accuracy: 0.4973
21472/25000 [========================>.....] - ETA: 9s - loss: 7.7059 - accuracy: 0.4974
21504/25000 [========================>.....] - ETA: 9s - loss: 7.7058 - accuracy: 0.4974
21536/25000 [========================>.....] - ETA: 8s - loss: 7.7015 - accuracy: 0.4977
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6993 - accuracy: 0.4979
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6971 - accuracy: 0.4980
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6992 - accuracy: 0.4979
21664/25000 [========================>.....] - ETA: 8s - loss: 7.7006 - accuracy: 0.4978
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6991 - accuracy: 0.4979
21728/25000 [=========================>....] - ETA: 8s - loss: 7.7012 - accuracy: 0.4977
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6997 - accuracy: 0.4978
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6976 - accuracy: 0.4980
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6968 - accuracy: 0.4980
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6947 - accuracy: 0.4982
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6946 - accuracy: 0.4982
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6967 - accuracy: 0.4980
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6939 - accuracy: 0.4982
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6917 - accuracy: 0.4984
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6931 - accuracy: 0.4983
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6924 - accuracy: 0.4983
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6930 - accuracy: 0.4983
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6916 - accuracy: 0.4984
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6915 - accuracy: 0.4984
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6874 - accuracy: 0.4986
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6866 - accuracy: 0.4987
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6852 - accuracy: 0.4988
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6831 - accuracy: 0.4989
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6810 - accuracy: 0.4991
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6824 - accuracy: 0.4990
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6824 - accuracy: 0.4990
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6817 - accuracy: 0.4990
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6823 - accuracy: 0.4990
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6809 - accuracy: 0.4991
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6789 - accuracy: 0.4992
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6795 - accuracy: 0.4992
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6816 - accuracy: 0.4990
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6795 - accuracy: 0.4992
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6781 - accuracy: 0.4992
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6828 - accuracy: 0.4989
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6828 - accuracy: 0.4989
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6835 - accuracy: 0.4989
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6828 - accuracy: 0.4989
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6821 - accuracy: 0.4990
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6794 - accuracy: 0.4992
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6834 - accuracy: 0.4989
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6827 - accuracy: 0.4990
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6807 - accuracy: 0.4991
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6813 - accuracy: 0.4990
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6793 - accuracy: 0.4992
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6773 - accuracy: 0.4993
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6759 - accuracy: 0.4994
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6786 - accuracy: 0.4992
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6792 - accuracy: 0.4992
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6772 - accuracy: 0.4993
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6759 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6778 - accuracy: 0.4993
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6798 - accuracy: 0.4991
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6785 - accuracy: 0.4992
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6765 - accuracy: 0.4994
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6771 - accuracy: 0.4993
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6758 - accuracy: 0.4994
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6751 - accuracy: 0.4994
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6770 - accuracy: 0.4993
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24192/25000 [============================>.] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24224/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24256/25000 [============================>.] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
24288/25000 [============================>.] - ETA: 1s - loss: 7.6571 - accuracy: 0.5006
24320/25000 [============================>.] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
24352/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24384/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24416/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24448/25000 [============================>.] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
24480/25000 [============================>.] - ETA: 1s - loss: 7.6560 - accuracy: 0.5007
24512/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24544/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24576/25000 [============================>.] - ETA: 1s - loss: 7.6623 - accuracy: 0.5003
24608/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24672/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24768/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24800/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24864/25000 [============================>.] - ETA: 0s - loss: 7.6605 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 76s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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


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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:51<01:16, 25.58s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.31689108984350134, 'embedding_size_factor': 0.6049314548339438, 'layers.choice': 1, 'learning_rate': 0.008300895251150976, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.139277827790618e-08} and reward: 0.3782
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4G\xf1\x90\xd1b|X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3[\x995\xda\xab\xf3X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x81\x00\x0fM\x0c\xba\x87X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>`\xda\x971\x1b\xe7\xcdu.' and reward: 0.3782
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4G\xf1\x90\xd1b|X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3[\x995\xda\xab\xf3X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x81\x00\x0fM\x0c\xba\x87X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>`\xda\x971\x1b\xe7\xcdu.' and reward: 0.3782
 60%|██████    | 3/5 [02:46<01:44, 52.47s/it] 60%|██████    | 3/5 [02:46<01:50, 55.46s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.32412181742620155, 'embedding_size_factor': 0.991843101474321, 'layers.choice': 0, 'learning_rate': 0.007861599719459537, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 5.298112127550989e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xbeiop\xff\xe5X\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xbd-\xbes\r\xaaX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?\x80\x19\xbe\r\x83\xba\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xd68\xcd\xf8\xed\x9fzu.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xbeiop\xff\xe5X\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xbd-\xbes\r\xaaX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?\x80\x19\xbe\r\x83\xba\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xd68\xcd\xf8\xed\x9fzu.' and reward: 0.3894
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 219.03660345077515
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 2, 'dropout_prob': 0.32412181742620155, 'embedding_size_factor': 0.991843101474321, 'layers.choice': 0, 'learning_rate': 0.007861599719459537, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 5.298112127550989e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -101.84s of remaining time.
Ensemble size: 63
Ensemble weights: 
[0.34920635 0.38095238 0.26984127]
	0.3942	 = Validation accuracy score
	1.09s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 222.97s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl





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
 2588672/17464789 [===>..........................] - ETA: 0s
10493952/17464789 [=================>............] - ETA: 0s
16228352/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-26 10:22:35.028216: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-26 10:22:35.033033: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-26 10:22:35.033239: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e0e3a48d00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 10:22:35.033255: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:09 - loss: 8.6249 - accuracy: 0.4375
   64/25000 [..............................] - ETA: 3:05 - loss: 6.9479 - accuracy: 0.5469
   96/25000 [..............................] - ETA: 2:23 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 2:02 - loss: 7.4270 - accuracy: 0.5156
  160/25000 [..............................] - ETA: 1:49 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:40 - loss: 7.4270 - accuracy: 0.5156
  224/25000 [..............................] - ETA: 1:34 - loss: 7.8720 - accuracy: 0.4866
  256/25000 [..............................] - ETA: 1:30 - loss: 7.7265 - accuracy: 0.4961
  288/25000 [..............................] - ETA: 1:26 - loss: 7.6134 - accuracy: 0.5035
  320/25000 [..............................] - ETA: 1:23 - loss: 7.5229 - accuracy: 0.5094
  352/25000 [..............................] - ETA: 1:21 - loss: 7.2310 - accuracy: 0.5284
  384/25000 [..............................] - ETA: 1:19 - loss: 7.1475 - accuracy: 0.5339
  416/25000 [..............................] - ETA: 1:18 - loss: 7.3717 - accuracy: 0.5192
  448/25000 [..............................] - ETA: 1:16 - loss: 7.4955 - accuracy: 0.5112
  480/25000 [..............................] - ETA: 1:15 - loss: 7.4111 - accuracy: 0.5167
  512/25000 [..............................] - ETA: 1:14 - loss: 7.5768 - accuracy: 0.5059
  544/25000 [..............................] - ETA: 1:13 - loss: 7.6666 - accuracy: 0.5000
  576/25000 [..............................] - ETA: 1:12 - loss: 7.7465 - accuracy: 0.4948
  608/25000 [..............................] - ETA: 1:11 - loss: 7.7675 - accuracy: 0.4934
  640/25000 [..............................] - ETA: 1:11 - loss: 7.8343 - accuracy: 0.4891
  672/25000 [..............................] - ETA: 1:10 - loss: 7.8948 - accuracy: 0.4851
  704/25000 [..............................] - ETA: 1:10 - loss: 7.8409 - accuracy: 0.4886
  736/25000 [..............................] - ETA: 1:09 - loss: 7.7916 - accuracy: 0.4918
  768/25000 [..............................] - ETA: 1:08 - loss: 7.6866 - accuracy: 0.4987
  800/25000 [..............................] - ETA: 1:08 - loss: 7.6666 - accuracy: 0.5000
  832/25000 [..............................] - ETA: 1:07 - loss: 7.6666 - accuracy: 0.5000
  864/25000 [>.............................] - ETA: 1:07 - loss: 7.6489 - accuracy: 0.5012
  896/25000 [>.............................] - ETA: 1:07 - loss: 7.6324 - accuracy: 0.5022
  928/25000 [>.............................] - ETA: 1:06 - loss: 7.5510 - accuracy: 0.5075
  960/25000 [>.............................] - ETA: 1:06 - loss: 7.5388 - accuracy: 0.5083
  992/25000 [>.............................] - ETA: 1:06 - loss: 7.5430 - accuracy: 0.5081
 1024/25000 [>.............................] - ETA: 1:05 - loss: 7.6067 - accuracy: 0.5039
 1056/25000 [>.............................] - ETA: 1:05 - loss: 7.6085 - accuracy: 0.5038
 1088/25000 [>.............................] - ETA: 1:05 - loss: 7.5962 - accuracy: 0.5046
 1120/25000 [>.............................] - ETA: 1:04 - loss: 7.5982 - accuracy: 0.5045
 1152/25000 [>.............................] - ETA: 1:04 - loss: 7.5868 - accuracy: 0.5052
 1184/25000 [>.............................] - ETA: 1:04 - loss: 7.6019 - accuracy: 0.5042
 1216/25000 [>.............................] - ETA: 1:04 - loss: 7.6540 - accuracy: 0.5008
 1248/25000 [>.............................] - ETA: 1:03 - loss: 7.6175 - accuracy: 0.5032
 1280/25000 [>.............................] - ETA: 1:03 - loss: 7.6307 - accuracy: 0.5023
 1312/25000 [>.............................] - ETA: 1:03 - loss: 7.6199 - accuracy: 0.5030
 1344/25000 [>.............................] - ETA: 1:03 - loss: 7.6096 - accuracy: 0.5037
 1376/25000 [>.............................] - ETA: 1:03 - loss: 7.5440 - accuracy: 0.5080
 1408/25000 [>.............................] - ETA: 1:02 - loss: 7.5468 - accuracy: 0.5078
 1440/25000 [>.............................] - ETA: 1:02 - loss: 7.5495 - accuracy: 0.5076
 1472/25000 [>.............................] - ETA: 1:02 - loss: 7.5520 - accuracy: 0.5075
 1504/25000 [>.............................] - ETA: 1:02 - loss: 7.5953 - accuracy: 0.5047
 1536/25000 [>.............................] - ETA: 1:01 - loss: 7.6067 - accuracy: 0.5039
 1568/25000 [>.............................] - ETA: 1:01 - loss: 7.6079 - accuracy: 0.5038
 1600/25000 [>.............................] - ETA: 1:01 - loss: 7.5420 - accuracy: 0.5081
 1632/25000 [>.............................] - ETA: 1:01 - loss: 7.5821 - accuracy: 0.5055
 1664/25000 [>.............................] - ETA: 1:01 - loss: 7.5745 - accuracy: 0.5060
 1696/25000 [=>............................] - ETA: 1:00 - loss: 7.6124 - accuracy: 0.5035
 1728/25000 [=>............................] - ETA: 1:00 - loss: 7.6134 - accuracy: 0.5035
 1760/25000 [=>............................] - ETA: 1:00 - loss: 7.5621 - accuracy: 0.5068
 1792/25000 [=>............................] - ETA: 1:00 - loss: 7.5383 - accuracy: 0.5084
 1824/25000 [=>............................] - ETA: 1:00 - loss: 7.5826 - accuracy: 0.5055
 1856/25000 [=>............................] - ETA: 1:00 - loss: 7.6171 - accuracy: 0.5032
 1888/25000 [=>............................] - ETA: 1:00 - loss: 7.6016 - accuracy: 0.5042
 1920/25000 [=>............................] - ETA: 59s - loss: 7.5947 - accuracy: 0.5047 
 1952/25000 [=>............................] - ETA: 59s - loss: 7.6431 - accuracy: 0.5015
 1984/25000 [=>............................] - ETA: 59s - loss: 7.6589 - accuracy: 0.5005
 2016/25000 [=>............................] - ETA: 59s - loss: 7.6438 - accuracy: 0.5015
 2048/25000 [=>............................] - ETA: 59s - loss: 7.6591 - accuracy: 0.5005
 2080/25000 [=>............................] - ETA: 59s - loss: 7.6887 - accuracy: 0.4986
 2112/25000 [=>............................] - ETA: 59s - loss: 7.7102 - accuracy: 0.4972
 2144/25000 [=>............................] - ETA: 59s - loss: 7.6952 - accuracy: 0.4981
 2176/25000 [=>............................] - ETA: 58s - loss: 7.7089 - accuracy: 0.4972
 2208/25000 [=>............................] - ETA: 58s - loss: 7.6805 - accuracy: 0.4991
 2240/25000 [=>............................] - ETA: 58s - loss: 7.6735 - accuracy: 0.4996
 2272/25000 [=>............................] - ETA: 58s - loss: 7.6801 - accuracy: 0.4991
 2304/25000 [=>............................] - ETA: 58s - loss: 7.6932 - accuracy: 0.4983
 2336/25000 [=>............................] - ETA: 58s - loss: 7.6732 - accuracy: 0.4996
 2368/25000 [=>............................] - ETA: 58s - loss: 7.6925 - accuracy: 0.4983
 2400/25000 [=>............................] - ETA: 58s - loss: 7.6730 - accuracy: 0.4996
 2432/25000 [=>............................] - ETA: 58s - loss: 7.6540 - accuracy: 0.5008
 2464/25000 [=>............................] - ETA: 58s - loss: 7.6542 - accuracy: 0.5008
 2496/25000 [=>............................] - ETA: 58s - loss: 7.6543 - accuracy: 0.5008
 2528/25000 [==>...........................] - ETA: 57s - loss: 7.6363 - accuracy: 0.5020
 2560/25000 [==>...........................] - ETA: 57s - loss: 7.6546 - accuracy: 0.5008
 2592/25000 [==>...........................] - ETA: 57s - loss: 7.6548 - accuracy: 0.5008
 2624/25000 [==>...........................] - ETA: 57s - loss: 7.6549 - accuracy: 0.5008
 2656/25000 [==>...........................] - ETA: 57s - loss: 7.6551 - accuracy: 0.5008
 2688/25000 [==>...........................] - ETA: 57s - loss: 7.6609 - accuracy: 0.5004
 2720/25000 [==>...........................] - ETA: 57s - loss: 7.6441 - accuracy: 0.5015
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.6555 - accuracy: 0.5007
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.6556 - accuracy: 0.5007
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.6503 - accuracy: 0.5011
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.6343 - accuracy: 0.5021
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.6240 - accuracy: 0.5028
 2912/25000 [==>...........................] - ETA: 56s - loss: 7.6192 - accuracy: 0.5031
 2944/25000 [==>...........................] - ETA: 56s - loss: 7.6093 - accuracy: 0.5037
 2976/25000 [==>...........................] - ETA: 56s - loss: 7.6409 - accuracy: 0.5017
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.6411 - accuracy: 0.5017
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.6515 - accuracy: 0.5010
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.6566 - accuracy: 0.5007
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.6567 - accuracy: 0.5006
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.6715 - accuracy: 0.4997
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.6810 - accuracy: 0.4991
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.6951 - accuracy: 0.4981
 3264/25000 [==>...........................] - ETA: 56s - loss: 7.6760 - accuracy: 0.4994
 3296/25000 [==>...........................] - ETA: 55s - loss: 7.6806 - accuracy: 0.4991
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.7081 - accuracy: 0.4973
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.6849 - accuracy: 0.4988
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.6937 - accuracy: 0.4982
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.6801 - accuracy: 0.4991
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.6755 - accuracy: 0.4994
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.6974 - accuracy: 0.4980
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.7058 - accuracy: 0.4974
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.6968 - accuracy: 0.4980
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.7051 - accuracy: 0.4975
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.7133 - accuracy: 0.4970
 3648/25000 [===>..........................] - ETA: 55s - loss: 7.7087 - accuracy: 0.4973
 3680/25000 [===>..........................] - ETA: 54s - loss: 7.7083 - accuracy: 0.4973
 3712/25000 [===>..........................] - ETA: 54s - loss: 7.6955 - accuracy: 0.4981
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.6707 - accuracy: 0.4997
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.6544 - accuracy: 0.5008
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.6505 - accuracy: 0.5011
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.6506 - accuracy: 0.5010
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.6429 - accuracy: 0.5015
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.6509 - accuracy: 0.5010
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.6471 - accuracy: 0.5013
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.6512 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 54s - loss: 7.6436 - accuracy: 0.5015
 4032/25000 [===>..........................] - ETA: 53s - loss: 7.6514 - accuracy: 0.5010
 4064/25000 [===>..........................] - ETA: 53s - loss: 7.6213 - accuracy: 0.5030
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.6367 - accuracy: 0.5020
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.6369 - accuracy: 0.5019
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.6482 - accuracy: 0.5012
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.6483 - accuracy: 0.5012
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.6339 - accuracy: 0.5021
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.6306 - accuracy: 0.5023
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.6273 - accuracy: 0.5026
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.6382 - accuracy: 0.5019
 4352/25000 [====>.........................] - ETA: 52s - loss: 7.6525 - accuracy: 0.5009
 4384/25000 [====>.........................] - ETA: 52s - loss: 7.6631 - accuracy: 0.5002
 4416/25000 [====>.........................] - ETA: 52s - loss: 7.6562 - accuracy: 0.5007
 4448/25000 [====>.........................] - ETA: 52s - loss: 7.6459 - accuracy: 0.5013
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.6529 - accuracy: 0.5009
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.6530 - accuracy: 0.5009
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.6430 - accuracy: 0.5015
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.6264 - accuracy: 0.5026
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.6200 - accuracy: 0.5030
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.6204 - accuracy: 0.5030
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.6043 - accuracy: 0.5041
 4704/25000 [====>.........................] - ETA: 52s - loss: 7.5949 - accuracy: 0.5047
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.5857 - accuracy: 0.5053
 4768/25000 [====>.........................] - ETA: 51s - loss: 7.5701 - accuracy: 0.5063
 4800/25000 [====>.........................] - ETA: 51s - loss: 7.5708 - accuracy: 0.5063
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.5714 - accuracy: 0.5062
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.5720 - accuracy: 0.5062
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.5727 - accuracy: 0.5061
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.5546 - accuracy: 0.5073
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.5399 - accuracy: 0.5083
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.5376 - accuracy: 0.5084
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.5384 - accuracy: 0.5084
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.5271 - accuracy: 0.5091
 5088/25000 [=====>........................] - ETA: 50s - loss: 7.5370 - accuracy: 0.5085
 5120/25000 [=====>........................] - ETA: 50s - loss: 7.5438 - accuracy: 0.5080
 5152/25000 [=====>........................] - ETA: 50s - loss: 7.5267 - accuracy: 0.5091
 5184/25000 [=====>........................] - ETA: 50s - loss: 7.5394 - accuracy: 0.5083
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.5314 - accuracy: 0.5088
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.5322 - accuracy: 0.5088
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.5359 - accuracy: 0.5085
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.5483 - accuracy: 0.5077
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.5432 - accuracy: 0.5080
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.5554 - accuracy: 0.5073
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.5504 - accuracy: 0.5076
 5440/25000 [=====>........................] - ETA: 49s - loss: 7.5482 - accuracy: 0.5077
 5472/25000 [=====>........................] - ETA: 49s - loss: 7.5517 - accuracy: 0.5075
 5504/25000 [=====>........................] - ETA: 49s - loss: 7.5385 - accuracy: 0.5084
 5536/25000 [=====>........................] - ETA: 49s - loss: 7.5392 - accuracy: 0.5083
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.5262 - accuracy: 0.5092
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.5160 - accuracy: 0.5098
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.5169 - accuracy: 0.5098
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.5177 - accuracy: 0.5097
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.5159 - accuracy: 0.5098
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.5221 - accuracy: 0.5094
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.5255 - accuracy: 0.5092
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.5237 - accuracy: 0.5093
 5824/25000 [=====>........................] - ETA: 48s - loss: 7.5297 - accuracy: 0.5089
 5856/25000 [======>.......................] - ETA: 48s - loss: 7.5331 - accuracy: 0.5087
 5888/25000 [======>.......................] - ETA: 48s - loss: 7.5364 - accuracy: 0.5085
 5920/25000 [======>.......................] - ETA: 48s - loss: 7.5371 - accuracy: 0.5084
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.5378 - accuracy: 0.5084
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.5359 - accuracy: 0.5085
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.5366 - accuracy: 0.5085
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.5424 - accuracy: 0.5081
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.5355 - accuracy: 0.5086
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.5362 - accuracy: 0.5085
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.5294 - accuracy: 0.5090
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.5127 - accuracy: 0.5100
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.5110 - accuracy: 0.5101
 6240/25000 [======>.......................] - ETA: 47s - loss: 7.5167 - accuracy: 0.5098
 6272/25000 [======>.......................] - ETA: 47s - loss: 7.5273 - accuracy: 0.5091
 6304/25000 [======>.......................] - ETA: 47s - loss: 7.5231 - accuracy: 0.5094
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.5166 - accuracy: 0.5098
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.5149 - accuracy: 0.5099
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.5205 - accuracy: 0.5095
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.5188 - accuracy: 0.5096
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.5101 - accuracy: 0.5102
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.5085 - accuracy: 0.5103
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.5233 - accuracy: 0.5093
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.5170 - accuracy: 0.5098
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.5178 - accuracy: 0.5097
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.5115 - accuracy: 0.5101
 6656/25000 [======>.......................] - ETA: 46s - loss: 7.5146 - accuracy: 0.5099
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.5176 - accuracy: 0.5097
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.5160 - accuracy: 0.5098
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.4986 - accuracy: 0.5110
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.4971 - accuracy: 0.5111
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.5024 - accuracy: 0.5107
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.5032 - accuracy: 0.5107
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.5128 - accuracy: 0.5100
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.5025 - accuracy: 0.5107
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.5054 - accuracy: 0.5105
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.5062 - accuracy: 0.5105
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.5047 - accuracy: 0.5106
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.4924 - accuracy: 0.5114
 7072/25000 [=======>......................] - ETA: 45s - loss: 7.4975 - accuracy: 0.5110
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.4961 - accuracy: 0.5111
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.4969 - accuracy: 0.5111
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.5062 - accuracy: 0.5105
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.5069 - accuracy: 0.5104
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.5140 - accuracy: 0.5100
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.5189 - accuracy: 0.5096
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.5111 - accuracy: 0.5101
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.5222 - accuracy: 0.5094
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.5250 - accuracy: 0.5092
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.5214 - accuracy: 0.5095
 7424/25000 [=======>......................] - ETA: 44s - loss: 7.5262 - accuracy: 0.5092
 7456/25000 [=======>......................] - ETA: 44s - loss: 7.5371 - accuracy: 0.5084
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.5397 - accuracy: 0.5083
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.5422 - accuracy: 0.5081
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.5529 - accuracy: 0.5074
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.5433 - accuracy: 0.5080
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.5478 - accuracy: 0.5077
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.5503 - accuracy: 0.5076
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.5528 - accuracy: 0.5074
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.5493 - accuracy: 0.5077
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.5518 - accuracy: 0.5075
 7776/25000 [========>.....................] - ETA: 43s - loss: 7.5523 - accuracy: 0.5075
 7808/25000 [========>.....................] - ETA: 43s - loss: 7.5488 - accuracy: 0.5077
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.5473 - accuracy: 0.5078
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.5439 - accuracy: 0.5080
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.5463 - accuracy: 0.5078
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.5449 - accuracy: 0.5079
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.5512 - accuracy: 0.5075
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.5478 - accuracy: 0.5077
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.5464 - accuracy: 0.5078
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.5430 - accuracy: 0.5081
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.5435 - accuracy: 0.5080
 8128/25000 [========>.....................] - ETA: 42s - loss: 7.5478 - accuracy: 0.5078
 8160/25000 [========>.....................] - ETA: 42s - loss: 7.5539 - accuracy: 0.5074
 8192/25000 [========>.....................] - ETA: 42s - loss: 7.5450 - accuracy: 0.5079
 8224/25000 [========>.....................] - ETA: 42s - loss: 7.5454 - accuracy: 0.5079
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.5440 - accuracy: 0.5080
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.5464 - accuracy: 0.5078
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.5395 - accuracy: 0.5083
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.5399 - accuracy: 0.5083
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.5441 - accuracy: 0.5080
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.5391 - accuracy: 0.5083
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.5414 - accuracy: 0.5082
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.5509 - accuracy: 0.5075
 8512/25000 [=========>....................] - ETA: 41s - loss: 7.5531 - accuracy: 0.5074
 8544/25000 [=========>....................] - ETA: 41s - loss: 7.5571 - accuracy: 0.5071
 8576/25000 [=========>....................] - ETA: 41s - loss: 7.5611 - accuracy: 0.5069
 8608/25000 [=========>....................] - ETA: 41s - loss: 7.5580 - accuracy: 0.5071
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.5584 - accuracy: 0.5071
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.5641 - accuracy: 0.5067
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.5609 - accuracy: 0.5069
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.5683 - accuracy: 0.5064
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.5582 - accuracy: 0.5071
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.5568 - accuracy: 0.5072
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.5555 - accuracy: 0.5072
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.5594 - accuracy: 0.5070
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.5667 - accuracy: 0.5065
 8928/25000 [=========>....................] - ETA: 40s - loss: 7.5636 - accuracy: 0.5067
 8960/25000 [=========>....................] - ETA: 40s - loss: 7.5674 - accuracy: 0.5065
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.5762 - accuracy: 0.5059
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.5834 - accuracy: 0.5054
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.5837 - accuracy: 0.5054
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.5856 - accuracy: 0.5053
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.5876 - accuracy: 0.5052
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.5946 - accuracy: 0.5047
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.5915 - accuracy: 0.5049
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.5967 - accuracy: 0.5046
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.5970 - accuracy: 0.5045
 9280/25000 [==========>...................] - ETA: 39s - loss: 7.5972 - accuracy: 0.5045
 9312/25000 [==========>...................] - ETA: 39s - loss: 7.5925 - accuracy: 0.5048
 9344/25000 [==========>...................] - ETA: 39s - loss: 7.5928 - accuracy: 0.5048
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.5898 - accuracy: 0.5050
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.5884 - accuracy: 0.5051
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.5935 - accuracy: 0.5048
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.5970 - accuracy: 0.5045
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.5924 - accuracy: 0.5048
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.5943 - accuracy: 0.5047
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.5977 - accuracy: 0.5045
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.5963 - accuracy: 0.5046
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.5950 - accuracy: 0.5047
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.5905 - accuracy: 0.5050
 9696/25000 [==========>...................] - ETA: 38s - loss: 7.5860 - accuracy: 0.5053
 9728/25000 [==========>...................] - ETA: 38s - loss: 7.5862 - accuracy: 0.5052
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.5849 - accuracy: 0.5053
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.5883 - accuracy: 0.5051
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.5979 - accuracy: 0.5045
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.5997 - accuracy: 0.5044
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.5968 - accuracy: 0.5046
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.5909 - accuracy: 0.5049
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6019 - accuracy: 0.5042
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.5960 - accuracy: 0.5046
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6008 - accuracy: 0.5043
10048/25000 [===========>..................] - ETA: 38s - loss: 7.5964 - accuracy: 0.5046
10080/25000 [===========>..................] - ETA: 37s - loss: 7.5997 - accuracy: 0.5044
10112/25000 [===========>..................] - ETA: 37s - loss: 7.5969 - accuracy: 0.5045
10144/25000 [===========>..................] - ETA: 37s - loss: 7.6001 - accuracy: 0.5043
10176/25000 [===========>..................] - ETA: 37s - loss: 7.5973 - accuracy: 0.5045
10208/25000 [===========>..................] - ETA: 37s - loss: 7.5960 - accuracy: 0.5046
10240/25000 [===========>..................] - ETA: 37s - loss: 7.5888 - accuracy: 0.5051
10272/25000 [===========>..................] - ETA: 37s - loss: 7.5905 - accuracy: 0.5050
10304/25000 [===========>..................] - ETA: 37s - loss: 7.5892 - accuracy: 0.5050
10336/25000 [===========>..................] - ETA: 37s - loss: 7.5880 - accuracy: 0.5051
10368/25000 [===========>..................] - ETA: 37s - loss: 7.5882 - accuracy: 0.5051
10400/25000 [===========>..................] - ETA: 37s - loss: 7.5929 - accuracy: 0.5048
10432/25000 [===========>..................] - ETA: 37s - loss: 7.6005 - accuracy: 0.5043
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6051 - accuracy: 0.5040
10496/25000 [===========>..................] - ETA: 36s - loss: 7.6096 - accuracy: 0.5037
10528/25000 [===========>..................] - ETA: 36s - loss: 7.6127 - accuracy: 0.5035
10560/25000 [===========>..................] - ETA: 36s - loss: 7.6129 - accuracy: 0.5035
10592/25000 [===========>..................] - ETA: 36s - loss: 7.6160 - accuracy: 0.5033
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6175 - accuracy: 0.5032
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6220 - accuracy: 0.5029
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6279 - accuracy: 0.5025
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6266 - accuracy: 0.5026
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6267 - accuracy: 0.5026
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6297 - accuracy: 0.5024
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6298 - accuracy: 0.5024
10848/25000 [============>.................] - ETA: 36s - loss: 7.6285 - accuracy: 0.5025
10880/25000 [============>.................] - ETA: 36s - loss: 7.6314 - accuracy: 0.5023
10912/25000 [============>.................] - ETA: 35s - loss: 7.6301 - accuracy: 0.5024
10944/25000 [============>.................] - ETA: 35s - loss: 7.6274 - accuracy: 0.5026
10976/25000 [============>.................] - ETA: 35s - loss: 7.6247 - accuracy: 0.5027
11008/25000 [============>.................] - ETA: 35s - loss: 7.6276 - accuracy: 0.5025
11040/25000 [============>.................] - ETA: 35s - loss: 7.6319 - accuracy: 0.5023
11072/25000 [============>.................] - ETA: 35s - loss: 7.6265 - accuracy: 0.5026
11104/25000 [============>.................] - ETA: 35s - loss: 7.6266 - accuracy: 0.5026
11136/25000 [============>.................] - ETA: 35s - loss: 7.6239 - accuracy: 0.5028
11168/25000 [============>.................] - ETA: 35s - loss: 7.6213 - accuracy: 0.5030
11200/25000 [============>.................] - ETA: 35s - loss: 7.6160 - accuracy: 0.5033
11232/25000 [============>.................] - ETA: 35s - loss: 7.6120 - accuracy: 0.5036
11264/25000 [============>.................] - ETA: 35s - loss: 7.6163 - accuracy: 0.5033
11296/25000 [============>.................] - ETA: 35s - loss: 7.6137 - accuracy: 0.5035
11328/25000 [============>.................] - ETA: 34s - loss: 7.6152 - accuracy: 0.5034
11360/25000 [============>.................] - ETA: 34s - loss: 7.6126 - accuracy: 0.5035
11392/25000 [============>.................] - ETA: 34s - loss: 7.6222 - accuracy: 0.5029
11424/25000 [============>.................] - ETA: 34s - loss: 7.6183 - accuracy: 0.5032
11456/25000 [============>.................] - ETA: 34s - loss: 7.6184 - accuracy: 0.5031
11488/25000 [============>.................] - ETA: 34s - loss: 7.6199 - accuracy: 0.5030
11520/25000 [============>.................] - ETA: 34s - loss: 7.6267 - accuracy: 0.5026
11552/25000 [============>.................] - ETA: 34s - loss: 7.6348 - accuracy: 0.5021
11584/25000 [============>.................] - ETA: 34s - loss: 7.6388 - accuracy: 0.5018
11616/25000 [============>.................] - ETA: 34s - loss: 7.6363 - accuracy: 0.5020
11648/25000 [============>.................] - ETA: 34s - loss: 7.6377 - accuracy: 0.5019
11680/25000 [=============>................] - ETA: 34s - loss: 7.6391 - accuracy: 0.5018
11712/25000 [=============>................] - ETA: 33s - loss: 7.6470 - accuracy: 0.5013
11744/25000 [=============>................] - ETA: 33s - loss: 7.6483 - accuracy: 0.5012
11776/25000 [=============>................] - ETA: 33s - loss: 7.6510 - accuracy: 0.5010
11808/25000 [=============>................] - ETA: 33s - loss: 7.6536 - accuracy: 0.5008
11840/25000 [=============>................] - ETA: 33s - loss: 7.6511 - accuracy: 0.5010
11872/25000 [=============>................] - ETA: 33s - loss: 7.6524 - accuracy: 0.5009
11904/25000 [=============>................] - ETA: 33s - loss: 7.6576 - accuracy: 0.5006
11936/25000 [=============>................] - ETA: 33s - loss: 7.6512 - accuracy: 0.5010
11968/25000 [=============>................] - ETA: 33s - loss: 7.6474 - accuracy: 0.5013
12000/25000 [=============>................] - ETA: 33s - loss: 7.6398 - accuracy: 0.5017
12032/25000 [=============>................] - ETA: 33s - loss: 7.6411 - accuracy: 0.5017
12064/25000 [=============>................] - ETA: 33s - loss: 7.6399 - accuracy: 0.5017
12096/25000 [=============>................] - ETA: 32s - loss: 7.6400 - accuracy: 0.5017
12128/25000 [=============>................] - ETA: 32s - loss: 7.6439 - accuracy: 0.5015
12160/25000 [=============>................] - ETA: 32s - loss: 7.6427 - accuracy: 0.5016
12192/25000 [=============>................] - ETA: 32s - loss: 7.6402 - accuracy: 0.5017
12224/25000 [=============>................] - ETA: 32s - loss: 7.6340 - accuracy: 0.5021
12256/25000 [=============>................] - ETA: 32s - loss: 7.6316 - accuracy: 0.5023
12288/25000 [=============>................] - ETA: 32s - loss: 7.6367 - accuracy: 0.5020
12320/25000 [=============>................] - ETA: 32s - loss: 7.6392 - accuracy: 0.5018
12352/25000 [=============>................] - ETA: 32s - loss: 7.6381 - accuracy: 0.5019
12384/25000 [=============>................] - ETA: 32s - loss: 7.6431 - accuracy: 0.5015
12416/25000 [=============>................] - ETA: 32s - loss: 7.6333 - accuracy: 0.5022
12448/25000 [=============>................] - ETA: 32s - loss: 7.6272 - accuracy: 0.5026
12480/25000 [=============>................] - ETA: 31s - loss: 7.6273 - accuracy: 0.5026
12512/25000 [==============>...............] - ETA: 31s - loss: 7.6250 - accuracy: 0.5027
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6263 - accuracy: 0.5026
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6325 - accuracy: 0.5022
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6314 - accuracy: 0.5023
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6339 - accuracy: 0.5021
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6279 - accuracy: 0.5025
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6268 - accuracy: 0.5026
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6353 - accuracy: 0.5020
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6342 - accuracy: 0.5021
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6319 - accuracy: 0.5023
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6248 - accuracy: 0.5027
12864/25000 [==============>...............] - ETA: 30s - loss: 7.6321 - accuracy: 0.5023
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6321 - accuracy: 0.5022
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6334 - accuracy: 0.5022
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6311 - accuracy: 0.5023
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6312 - accuracy: 0.5023
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6301 - accuracy: 0.5024
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6314 - accuracy: 0.5023
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6315 - accuracy: 0.5023
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6269 - accuracy: 0.5026
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6281 - accuracy: 0.5025
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6259 - accuracy: 0.5027
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6260 - accuracy: 0.5026
13248/25000 [==============>...............] - ETA: 30s - loss: 7.6307 - accuracy: 0.5023
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6320 - accuracy: 0.5023
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6332 - accuracy: 0.5022
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6333 - accuracy: 0.5022
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6391 - accuracy: 0.5018
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6449 - accuracy: 0.5014
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6472 - accuracy: 0.5013
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6427 - accuracy: 0.5016
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6462 - accuracy: 0.5013
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6474 - accuracy: 0.5013
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6531 - accuracy: 0.5009
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6542 - accuracy: 0.5008
13632/25000 [===============>..............] - ETA: 29s - loss: 7.6531 - accuracy: 0.5009
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6509 - accuracy: 0.5010
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6509 - accuracy: 0.5010
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6476 - accuracy: 0.5012
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6443 - accuracy: 0.5015
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6422 - accuracy: 0.5016
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6422 - accuracy: 0.5016
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6423 - accuracy: 0.5016
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6434 - accuracy: 0.5015
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6446 - accuracy: 0.5014
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6468 - accuracy: 0.5013
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6513 - accuracy: 0.5010
14016/25000 [===============>..............] - ETA: 28s - loss: 7.6535 - accuracy: 0.5009
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6513 - accuracy: 0.5010
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6579 - accuracy: 0.5006
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6579 - accuracy: 0.5006
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6579 - accuracy: 0.5006
14176/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14208/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14240/25000 [================>.............] - ETA: 27s - loss: 7.6645 - accuracy: 0.5001
14272/25000 [================>.............] - ETA: 27s - loss: 7.6612 - accuracy: 0.5004
14304/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14336/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14368/25000 [================>.............] - ETA: 27s - loss: 7.6602 - accuracy: 0.5004
14400/25000 [================>.............] - ETA: 27s - loss: 7.6592 - accuracy: 0.5005
14432/25000 [================>.............] - ETA: 27s - loss: 7.6592 - accuracy: 0.5005
14464/25000 [================>.............] - ETA: 26s - loss: 7.6592 - accuracy: 0.5005
14496/25000 [================>.............] - ETA: 26s - loss: 7.6582 - accuracy: 0.5006
14528/25000 [================>.............] - ETA: 26s - loss: 7.6550 - accuracy: 0.5008
14560/25000 [================>.............] - ETA: 26s - loss: 7.6540 - accuracy: 0.5008
14592/25000 [================>.............] - ETA: 26s - loss: 7.6561 - accuracy: 0.5007
14624/25000 [================>.............] - ETA: 26s - loss: 7.6551 - accuracy: 0.5008
14656/25000 [================>.............] - ETA: 26s - loss: 7.6562 - accuracy: 0.5007
14688/25000 [================>.............] - ETA: 26s - loss: 7.6604 - accuracy: 0.5004
14720/25000 [================>.............] - ETA: 26s - loss: 7.6541 - accuracy: 0.5008
14752/25000 [================>.............] - ETA: 26s - loss: 7.6562 - accuracy: 0.5007
14784/25000 [================>.............] - ETA: 26s - loss: 7.6542 - accuracy: 0.5008
14816/25000 [================>.............] - ETA: 26s - loss: 7.6563 - accuracy: 0.5007
14848/25000 [================>.............] - ETA: 25s - loss: 7.6501 - accuracy: 0.5011
14880/25000 [================>.............] - ETA: 25s - loss: 7.6501 - accuracy: 0.5011
14912/25000 [================>.............] - ETA: 25s - loss: 7.6533 - accuracy: 0.5009
14944/25000 [================>.............] - ETA: 25s - loss: 7.6502 - accuracy: 0.5011
14976/25000 [================>.............] - ETA: 25s - loss: 7.6513 - accuracy: 0.5010
15008/25000 [=================>............] - ETA: 25s - loss: 7.6523 - accuracy: 0.5009
15040/25000 [=================>............] - ETA: 25s - loss: 7.6523 - accuracy: 0.5009
15072/25000 [=================>............] - ETA: 25s - loss: 7.6575 - accuracy: 0.5006
15104/25000 [=================>............] - ETA: 25s - loss: 7.6585 - accuracy: 0.5005
15136/25000 [=================>............] - ETA: 25s - loss: 7.6605 - accuracy: 0.5004
15168/25000 [=================>............] - ETA: 25s - loss: 7.6575 - accuracy: 0.5006
15200/25000 [=================>............] - ETA: 25s - loss: 7.6585 - accuracy: 0.5005
15232/25000 [=================>............] - ETA: 25s - loss: 7.6586 - accuracy: 0.5005
15264/25000 [=================>............] - ETA: 24s - loss: 7.6596 - accuracy: 0.5005
15296/25000 [=================>............] - ETA: 24s - loss: 7.6636 - accuracy: 0.5002
15328/25000 [=================>............] - ETA: 24s - loss: 7.6626 - accuracy: 0.5003
15360/25000 [=================>............] - ETA: 24s - loss: 7.6606 - accuracy: 0.5004
15392/25000 [=================>............] - ETA: 24s - loss: 7.6626 - accuracy: 0.5003
15424/25000 [=================>............] - ETA: 24s - loss: 7.6597 - accuracy: 0.5005
15456/25000 [=================>............] - ETA: 24s - loss: 7.6646 - accuracy: 0.5001
15488/25000 [=================>............] - ETA: 24s - loss: 7.6676 - accuracy: 0.4999
15520/25000 [=================>............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
15552/25000 [=================>............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
15584/25000 [=================>............] - ETA: 24s - loss: 7.6706 - accuracy: 0.4997
15616/25000 [=================>............] - ETA: 24s - loss: 7.6676 - accuracy: 0.4999
15648/25000 [=================>............] - ETA: 23s - loss: 7.6647 - accuracy: 0.5001
15680/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15712/25000 [=================>............] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
15744/25000 [=================>............] - ETA: 23s - loss: 7.6637 - accuracy: 0.5002
15776/25000 [=================>............] - ETA: 23s - loss: 7.6598 - accuracy: 0.5004
15808/25000 [=================>............] - ETA: 23s - loss: 7.6589 - accuracy: 0.5005
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6637 - accuracy: 0.5002
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6618 - accuracy: 0.5003
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6608 - accuracy: 0.5004
15936/25000 [==================>...........] - ETA: 23s - loss: 7.6618 - accuracy: 0.5003
15968/25000 [==================>...........] - ETA: 23s - loss: 7.6657 - accuracy: 0.5001
16000/25000 [==================>...........] - ETA: 23s - loss: 7.6657 - accuracy: 0.5001
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6685 - accuracy: 0.4999
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6628 - accuracy: 0.5002
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6647 - accuracy: 0.5001
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6685 - accuracy: 0.4999
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6723 - accuracy: 0.4996
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6723 - accuracy: 0.4996
16320/25000 [==================>...........] - ETA: 22s - loss: 7.6704 - accuracy: 0.4998
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6685 - accuracy: 0.4999
16384/25000 [==================>...........] - ETA: 22s - loss: 7.6657 - accuracy: 0.5001
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6629 - accuracy: 0.5002
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6638 - accuracy: 0.5002
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6675 - accuracy: 0.4999
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6638 - accuracy: 0.5002
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6629 - accuracy: 0.5002
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6620 - accuracy: 0.5003
16704/25000 [===================>..........] - ETA: 21s - loss: 7.6602 - accuracy: 0.5004
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6593 - accuracy: 0.5005
16768/25000 [===================>..........] - ETA: 21s - loss: 7.6566 - accuracy: 0.5007
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6575 - accuracy: 0.5006
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6566 - accuracy: 0.5007
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6548 - accuracy: 0.5008
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6521 - accuracy: 0.5009
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6512 - accuracy: 0.5010
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6549 - accuracy: 0.5008
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6576 - accuracy: 0.5006
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6567 - accuracy: 0.5006
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6603 - accuracy: 0.5004
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6585 - accuracy: 0.5005
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6612 - accuracy: 0.5004
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6586 - accuracy: 0.5005
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6604 - accuracy: 0.5004
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6604 - accuracy: 0.5004
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6604 - accuracy: 0.5004
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6622 - accuracy: 0.5003
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6640 - accuracy: 0.5002
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6684 - accuracy: 0.4999
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6640 - accuracy: 0.5002
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6649 - accuracy: 0.5001
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6675 - accuracy: 0.4999
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6692 - accuracy: 0.4998
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6701 - accuracy: 0.4998
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6701 - accuracy: 0.4998
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6744 - accuracy: 0.4995
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6727 - accuracy: 0.4996
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6770 - accuracy: 0.4993
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6752 - accuracy: 0.4994
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6744 - accuracy: 0.4995
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6735 - accuracy: 0.4996
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6718 - accuracy: 0.4997
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6718 - accuracy: 0.4997
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6675 - accuracy: 0.4999
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6675 - accuracy: 0.4999
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6675 - accuracy: 0.4999
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6683 - accuracy: 0.4999
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6692 - accuracy: 0.4998
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6658 - accuracy: 0.5001
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6641 - accuracy: 0.5002
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6683 - accuracy: 0.4999
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6717 - accuracy: 0.4997
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6700 - accuracy: 0.4998
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6733 - accuracy: 0.4996
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6742 - accuracy: 0.4995
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6733 - accuracy: 0.4996
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6700 - accuracy: 0.4998
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6725 - accuracy: 0.4996
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6732 - accuracy: 0.4996
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6724 - accuracy: 0.4996
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6699 - accuracy: 0.4998
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6625 - accuracy: 0.5003
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6601 - accuracy: 0.5004
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6601 - accuracy: 0.5004
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6601 - accuracy: 0.5004
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6585 - accuracy: 0.5005
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6569 - accuracy: 0.5006
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6577 - accuracy: 0.5006
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6569 - accuracy: 0.5006
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6561 - accuracy: 0.5007
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6561 - accuracy: 0.5007
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6561 - accuracy: 0.5007
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6529 - accuracy: 0.5009
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6546 - accuracy: 0.5008
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6554 - accuracy: 0.5007
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6570 - accuracy: 0.5006
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6522 - accuracy: 0.5009
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6562 - accuracy: 0.5007
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6618 - accuracy: 0.5003
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6603 - accuracy: 0.5004
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6627 - accuracy: 0.5003
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6658 - accuracy: 0.5001
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6603 - accuracy: 0.5004
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6603 - accuracy: 0.5004
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6580 - accuracy: 0.5006
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6595 - accuracy: 0.5005
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6603 - accuracy: 0.5004
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6549 - accuracy: 0.5008
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6525 - accuracy: 0.5009
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6541 - accuracy: 0.5008
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6596 - accuracy: 0.5005
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6612 - accuracy: 0.5004
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6588 - accuracy: 0.5005
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6620 - accuracy: 0.5003
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6604 - accuracy: 0.5004
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6612 - accuracy: 0.5004
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6604 - accuracy: 0.5004
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6604 - accuracy: 0.5004
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6620 - accuracy: 0.5003
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6628 - accuracy: 0.5003
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6589 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6551 - accuracy: 0.5008
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6582 - accuracy: 0.5005
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6590 - accuracy: 0.5005
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6628 - accuracy: 0.5002
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6628 - accuracy: 0.5002
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6643 - accuracy: 0.5001
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6605 - accuracy: 0.5004
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6621 - accuracy: 0.5003
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6606 - accuracy: 0.5004
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6636 - accuracy: 0.5002
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6644 - accuracy: 0.5001
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6629 - accuracy: 0.5002
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6651 - accuracy: 0.5001
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6644 - accuracy: 0.5001
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6637 - accuracy: 0.5002
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6629 - accuracy: 0.5002
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6629 - accuracy: 0.5002
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6637 - accuracy: 0.5002
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6637 - accuracy: 0.5002
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6607 - accuracy: 0.5004
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6593 - accuracy: 0.5005
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6600 - accuracy: 0.5004
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6600 - accuracy: 0.5004
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6608 - accuracy: 0.5004
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6593 - accuracy: 0.5005
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6586 - accuracy: 0.5005
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001 
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6695 - accuracy: 0.4998
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6716 - accuracy: 0.4997
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6709 - accuracy: 0.4997
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6694 - accuracy: 0.4998
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6709 - accuracy: 0.4997
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6687 - accuracy: 0.4999
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6673 - accuracy: 0.5000
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6638 - accuracy: 0.5002
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6617 - accuracy: 0.5003
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6604 - accuracy: 0.5004
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6618 - accuracy: 0.5003
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6631 - accuracy: 0.5002
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6625 - accuracy: 0.5003
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6638 - accuracy: 0.5002
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6632 - accuracy: 0.5002
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6659 - accuracy: 0.5000
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6639 - accuracy: 0.5002
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6639 - accuracy: 0.5002
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6611 - accuracy: 0.5004
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6550 - accuracy: 0.5008
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6577 - accuracy: 0.5006
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6598 - accuracy: 0.5004
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6605 - accuracy: 0.5004
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6598 - accuracy: 0.5004
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6632 - accuracy: 0.5002
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6572 - accuracy: 0.5006
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6606 - accuracy: 0.5004
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6699 - accuracy: 0.4998
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6628 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24224/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24256/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24288/25000 [============================>.] - ETA: 1s - loss: 7.6628 - accuracy: 0.5002
24320/25000 [============================>.] - ETA: 1s - loss: 7.6628 - accuracy: 0.5002
24352/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24448/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24544/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24576/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24608/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24704/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24736/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24800/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24832/25000 [============================>.] - ETA: 0s - loss: 7.6567 - accuracy: 0.5006
24864/25000 [============================>.] - ETA: 0s - loss: 7.6605 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 75s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

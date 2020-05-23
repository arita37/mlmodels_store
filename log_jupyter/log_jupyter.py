
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '3aee4395159545a95b0d7c8ed6830ec48eff1164', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3aee4395159545a95b0d7c8ed6830ec48eff1164

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164

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
	Data preprocessing and feature engineering runtime = 0.19s ...
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:42<01:04, 21.37s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.30057709977554825, 'embedding_size_factor': 0.7875981065189284, 'layers.choice': 2, 'learning_rate': 0.00015060735349766008, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 2.107744948841748e-12} and reward: 0.3768
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd3<\xa7\xbb]\x99\xbeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe94\x00\xf1\xbc}\x84X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?#\xbd\x8bP\xccj1X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x82\x8a8=&\xc8\xd7u.' and reward: 0.3768
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd3<\xa7\xbb]\x99\xbeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe94\x00\xf1\xbc}\x84X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?#\xbd\x8bP\xccj1X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x82\x8a8=&\xc8\xd7u.' and reward: 0.3768
 60%|██████    | 3/5 [01:25<00:55, 27.71s/it] 60%|██████    | 3/5 [01:25<00:56, 28.42s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.23285865736317618, 'embedding_size_factor': 0.918905383035475, 'layers.choice': 3, 'learning_rate': 0.00015871593052301175, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.08469294062621198} and reward: 0.3682
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\xceO\xfe\xfb\x8fRX\x15\x00\x00\x00embedding_size_factorq\x03G?\xedg\xacC\x08.\xd8X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?$\xcd\x9fv?&PX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xb5\xaeo\xc21\x0f\xfdu.' and reward: 0.3682
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\xceO\xfe\xfb\x8fRX\x15\x00\x00\x00embedding_size_factorq\x03G?\xedg\xacC\x08.\xd8X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?$\xcd\x9fv?&PX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xb5\xaeo\xc21\x0f\xfdu.' and reward: 0.3682
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 131.80946373939514
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.81s of the -13.97s of remaining time.
Ensemble size: 38
Ensemble weights: 
[0.5        0.26315789 0.23684211]
	0.3888	 = Validation accuracy score
	0.83s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 134.84s ...
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
 2842624/17464789 [===>..........................] - ETA: 0s
10625024/17464789 [=================>............] - ETA: 0s
16130048/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-23 05:19:09.310304: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 05:19:09.313824: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-23 05:19:09.313957: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560de9175bd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 05:19:09.313969: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 3:40 - loss: 6.7083 - accuracy: 0.5625
   64/25000 [..............................] - ETA: 2:17 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 1:48 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 1:33 - loss: 7.6666 - accuracy: 0.5000
  160/25000 [..............................] - ETA: 1:24 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:19 - loss: 7.3472 - accuracy: 0.5208
  224/25000 [..............................] - ETA: 1:16 - loss: 7.2559 - accuracy: 0.5268
  256/25000 [..............................] - ETA: 1:12 - loss: 7.3671 - accuracy: 0.5195
  288/25000 [..............................] - ETA: 1:10 - loss: 7.4004 - accuracy: 0.5174
  320/25000 [..............................] - ETA: 1:08 - loss: 7.4750 - accuracy: 0.5125
  352/25000 [..............................] - ETA: 1:06 - loss: 7.2746 - accuracy: 0.5256
  384/25000 [..............................] - ETA: 1:04 - loss: 7.1475 - accuracy: 0.5339
  416/25000 [..............................] - ETA: 1:03 - loss: 7.1506 - accuracy: 0.5337
  448/25000 [..............................] - ETA: 1:02 - loss: 7.1875 - accuracy: 0.5312
  480/25000 [..............................] - ETA: 1:01 - loss: 7.3152 - accuracy: 0.5229
  512/25000 [..............................] - ETA: 1:00 - loss: 7.4270 - accuracy: 0.5156
  544/25000 [..............................] - ETA: 1:00 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 59s - loss: 7.5601 - accuracy: 0.5069 
  608/25000 [..............................] - ETA: 59s - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 59s - loss: 7.5468 - accuracy: 0.5078
  672/25000 [..............................] - ETA: 58s - loss: 7.6210 - accuracy: 0.5030
  704/25000 [..............................] - ETA: 58s - loss: 7.6884 - accuracy: 0.4986
  736/25000 [..............................] - ETA: 58s - loss: 7.6458 - accuracy: 0.5014
  768/25000 [..............................] - ETA: 57s - loss: 7.7465 - accuracy: 0.4948
  800/25000 [..............................] - ETA: 57s - loss: 7.7625 - accuracy: 0.4938
  832/25000 [..............................] - ETA: 57s - loss: 7.7588 - accuracy: 0.4940
  864/25000 [>.............................] - ETA: 56s - loss: 7.8441 - accuracy: 0.4884
  896/25000 [>.............................] - ETA: 56s - loss: 7.7864 - accuracy: 0.4922
  928/25000 [>.............................] - ETA: 55s - loss: 7.8484 - accuracy: 0.4881
  960/25000 [>.............................] - ETA: 55s - loss: 7.8104 - accuracy: 0.4906
  992/25000 [>.............................] - ETA: 55s - loss: 7.8057 - accuracy: 0.4909
 1024/25000 [>.............................] - ETA: 55s - loss: 7.8164 - accuracy: 0.4902
 1056/25000 [>.............................] - ETA: 54s - loss: 7.7828 - accuracy: 0.4924
 1088/25000 [>.............................] - ETA: 54s - loss: 7.7794 - accuracy: 0.4926
 1120/25000 [>.............................] - ETA: 54s - loss: 7.8172 - accuracy: 0.4902
 1152/25000 [>.............................] - ETA: 54s - loss: 7.7997 - accuracy: 0.4913
 1184/25000 [>.............................] - ETA: 54s - loss: 7.7961 - accuracy: 0.4916
 1216/25000 [>.............................] - ETA: 54s - loss: 7.8305 - accuracy: 0.4893
 1248/25000 [>.............................] - ETA: 53s - loss: 7.8755 - accuracy: 0.4864
 1280/25000 [>.............................] - ETA: 53s - loss: 7.9182 - accuracy: 0.4836
 1312/25000 [>.............................] - ETA: 53s - loss: 7.9120 - accuracy: 0.4840
 1344/25000 [>.............................] - ETA: 53s - loss: 7.8948 - accuracy: 0.4851
 1376/25000 [>.............................] - ETA: 53s - loss: 7.9006 - accuracy: 0.4847
 1408/25000 [>.............................] - ETA: 52s - loss: 7.8844 - accuracy: 0.4858
 1440/25000 [>.............................] - ETA: 52s - loss: 7.8583 - accuracy: 0.4875
 1472/25000 [>.............................] - ETA: 52s - loss: 7.8020 - accuracy: 0.4912
 1504/25000 [>.............................] - ETA: 52s - loss: 7.7890 - accuracy: 0.4920
 1536/25000 [>.............................] - ETA: 52s - loss: 7.8363 - accuracy: 0.4889
 1568/25000 [>.............................] - ETA: 51s - loss: 7.8329 - accuracy: 0.4892
 1600/25000 [>.............................] - ETA: 51s - loss: 7.8008 - accuracy: 0.4913
 1632/25000 [>.............................] - ETA: 51s - loss: 7.8169 - accuracy: 0.4902
 1664/25000 [>.............................] - ETA: 51s - loss: 7.8417 - accuracy: 0.4886
 1696/25000 [=>............................] - ETA: 51s - loss: 7.8022 - accuracy: 0.4912
 1728/25000 [=>............................] - ETA: 51s - loss: 7.7820 - accuracy: 0.4925
 1760/25000 [=>............................] - ETA: 51s - loss: 7.7363 - accuracy: 0.4955
 1792/25000 [=>............................] - ETA: 51s - loss: 7.7436 - accuracy: 0.4950
 1824/25000 [=>............................] - ETA: 50s - loss: 7.7591 - accuracy: 0.4940
 1856/25000 [=>............................] - ETA: 50s - loss: 7.7658 - accuracy: 0.4935
 1888/25000 [=>............................] - ETA: 50s - loss: 7.7316 - accuracy: 0.4958
 1920/25000 [=>............................] - ETA: 50s - loss: 7.7225 - accuracy: 0.4964
 1952/25000 [=>............................] - ETA: 50s - loss: 7.7452 - accuracy: 0.4949
 1984/25000 [=>............................] - ETA: 50s - loss: 7.7825 - accuracy: 0.4924
 2016/25000 [=>............................] - ETA: 50s - loss: 7.7883 - accuracy: 0.4921
 2048/25000 [=>............................] - ETA: 50s - loss: 7.7639 - accuracy: 0.4937
 2080/25000 [=>............................] - ETA: 49s - loss: 7.7477 - accuracy: 0.4947
 2112/25000 [=>............................] - ETA: 49s - loss: 7.7392 - accuracy: 0.4953
 2144/25000 [=>............................] - ETA: 49s - loss: 7.7453 - accuracy: 0.4949
 2176/25000 [=>............................] - ETA: 49s - loss: 7.7653 - accuracy: 0.4936
 2208/25000 [=>............................] - ETA: 49s - loss: 7.7500 - accuracy: 0.4946
 2240/25000 [=>............................] - ETA: 49s - loss: 7.7693 - accuracy: 0.4933
 2272/25000 [=>............................] - ETA: 49s - loss: 7.7611 - accuracy: 0.4938
 2304/25000 [=>............................] - ETA: 49s - loss: 7.7798 - accuracy: 0.4926
 2336/25000 [=>............................] - ETA: 49s - loss: 7.7519 - accuracy: 0.4944
 2368/25000 [=>............................] - ETA: 48s - loss: 7.7702 - accuracy: 0.4932
 2400/25000 [=>............................] - ETA: 48s - loss: 7.7561 - accuracy: 0.4942
 2432/25000 [=>............................] - ETA: 48s - loss: 7.7486 - accuracy: 0.4947
 2464/25000 [=>............................] - ETA: 48s - loss: 7.7662 - accuracy: 0.4935
 2496/25000 [=>............................] - ETA: 48s - loss: 7.7772 - accuracy: 0.4928
 2528/25000 [==>...........................] - ETA: 48s - loss: 7.7515 - accuracy: 0.4945
 2560/25000 [==>...........................] - ETA: 48s - loss: 7.7385 - accuracy: 0.4953
 2592/25000 [==>...........................] - ETA: 48s - loss: 7.7317 - accuracy: 0.4958
 2624/25000 [==>...........................] - ETA: 48s - loss: 7.7484 - accuracy: 0.4947
 2656/25000 [==>...........................] - ETA: 48s - loss: 7.7705 - accuracy: 0.4932
 2688/25000 [==>...........................] - ETA: 48s - loss: 7.7807 - accuracy: 0.4926
 2720/25000 [==>...........................] - ETA: 48s - loss: 7.7794 - accuracy: 0.4926
 2752/25000 [==>...........................] - ETA: 47s - loss: 7.8003 - accuracy: 0.4913
 2784/25000 [==>...........................] - ETA: 47s - loss: 7.7823 - accuracy: 0.4925
 2816/25000 [==>...........................] - ETA: 47s - loss: 7.7864 - accuracy: 0.4922
 2848/25000 [==>...........................] - ETA: 47s - loss: 7.7743 - accuracy: 0.4930
 2880/25000 [==>...........................] - ETA: 47s - loss: 7.7784 - accuracy: 0.4927
 2912/25000 [==>...........................] - ETA: 47s - loss: 7.7614 - accuracy: 0.4938
 2944/25000 [==>...........................] - ETA: 47s - loss: 7.7760 - accuracy: 0.4929
 2976/25000 [==>...........................] - ETA: 47s - loss: 7.7697 - accuracy: 0.4933
 3008/25000 [==>...........................] - ETA: 47s - loss: 7.7635 - accuracy: 0.4937
 3040/25000 [==>...........................] - ETA: 47s - loss: 7.7675 - accuracy: 0.4934
 3072/25000 [==>...........................] - ETA: 47s - loss: 7.7615 - accuracy: 0.4938
 3104/25000 [==>...........................] - ETA: 47s - loss: 7.7407 - accuracy: 0.4952
 3136/25000 [==>...........................] - ETA: 47s - loss: 7.7302 - accuracy: 0.4959
 3168/25000 [==>...........................] - ETA: 47s - loss: 7.7295 - accuracy: 0.4959
 3200/25000 [==>...........................] - ETA: 47s - loss: 7.7289 - accuracy: 0.4959
 3232/25000 [==>...........................] - ETA: 46s - loss: 7.7046 - accuracy: 0.4975
 3264/25000 [==>...........................] - ETA: 46s - loss: 7.7324 - accuracy: 0.4957
 3296/25000 [==>...........................] - ETA: 46s - loss: 7.7364 - accuracy: 0.4954
 3328/25000 [==>...........................] - ETA: 46s - loss: 7.7496 - accuracy: 0.4946
 3360/25000 [===>..........................] - ETA: 46s - loss: 7.7442 - accuracy: 0.4949
 3392/25000 [===>..........................] - ETA: 46s - loss: 7.7525 - accuracy: 0.4944
 3424/25000 [===>..........................] - ETA: 46s - loss: 7.7562 - accuracy: 0.4942
 3456/25000 [===>..........................] - ETA: 46s - loss: 7.7420 - accuracy: 0.4951
 3488/25000 [===>..........................] - ETA: 46s - loss: 7.7326 - accuracy: 0.4957
 3520/25000 [===>..........................] - ETA: 46s - loss: 7.7232 - accuracy: 0.4963
 3552/25000 [===>..........................] - ETA: 46s - loss: 7.7141 - accuracy: 0.4969
 3584/25000 [===>..........................] - ETA: 46s - loss: 7.7180 - accuracy: 0.4967
 3616/25000 [===>..........................] - ETA: 45s - loss: 7.7005 - accuracy: 0.4978
 3648/25000 [===>..........................] - ETA: 45s - loss: 7.7044 - accuracy: 0.4975
 3680/25000 [===>..........................] - ETA: 45s - loss: 7.6916 - accuracy: 0.4984
 3712/25000 [===>..........................] - ETA: 45s - loss: 7.6873 - accuracy: 0.4987
 3744/25000 [===>..........................] - ETA: 45s - loss: 7.6707 - accuracy: 0.4997
 3776/25000 [===>..........................] - ETA: 45s - loss: 7.6626 - accuracy: 0.5003
 3808/25000 [===>..........................] - ETA: 45s - loss: 7.6626 - accuracy: 0.5003
 3840/25000 [===>..........................] - ETA: 45s - loss: 7.6706 - accuracy: 0.4997
 3872/25000 [===>..........................] - ETA: 45s - loss: 7.6864 - accuracy: 0.4987
 3904/25000 [===>..........................] - ETA: 45s - loss: 7.7059 - accuracy: 0.4974
 3936/25000 [===>..........................] - ETA: 45s - loss: 7.7056 - accuracy: 0.4975
 3968/25000 [===>..........................] - ETA: 45s - loss: 7.7284 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 45s - loss: 7.7356 - accuracy: 0.4955
 4032/25000 [===>..........................] - ETA: 45s - loss: 7.7351 - accuracy: 0.4955
 4064/25000 [===>..........................] - ETA: 44s - loss: 7.7459 - accuracy: 0.4948
 4096/25000 [===>..........................] - ETA: 44s - loss: 7.7527 - accuracy: 0.4944
 4128/25000 [===>..........................] - ETA: 44s - loss: 7.7706 - accuracy: 0.4932
 4160/25000 [===>..........................] - ETA: 44s - loss: 7.7698 - accuracy: 0.4933
 4192/25000 [====>.........................] - ETA: 44s - loss: 7.7507 - accuracy: 0.4945
 4224/25000 [====>.........................] - ETA: 44s - loss: 7.7465 - accuracy: 0.4948
 4256/25000 [====>.........................] - ETA: 44s - loss: 7.7279 - accuracy: 0.4960
 4288/25000 [====>.........................] - ETA: 44s - loss: 7.7274 - accuracy: 0.4960
 4320/25000 [====>.........................] - ETA: 44s - loss: 7.7305 - accuracy: 0.4958
 4352/25000 [====>.........................] - ETA: 44s - loss: 7.7300 - accuracy: 0.4959
 4384/25000 [====>.........................] - ETA: 44s - loss: 7.7401 - accuracy: 0.4952
 4416/25000 [====>.........................] - ETA: 44s - loss: 7.7361 - accuracy: 0.4955
 4448/25000 [====>.........................] - ETA: 43s - loss: 7.7321 - accuracy: 0.4957
 4480/25000 [====>.........................] - ETA: 43s - loss: 7.7385 - accuracy: 0.4953
 4512/25000 [====>.........................] - ETA: 43s - loss: 7.7278 - accuracy: 0.4960
 4544/25000 [====>.........................] - ETA: 43s - loss: 7.7274 - accuracy: 0.4960
 4576/25000 [====>.........................] - ETA: 43s - loss: 7.7102 - accuracy: 0.4972
 4608/25000 [====>.........................] - ETA: 43s - loss: 7.7065 - accuracy: 0.4974
 4640/25000 [====>.........................] - ETA: 43s - loss: 7.6997 - accuracy: 0.4978
 4672/25000 [====>.........................] - ETA: 43s - loss: 7.6962 - accuracy: 0.4981
 4704/25000 [====>.........................] - ETA: 43s - loss: 7.7057 - accuracy: 0.4974
 4736/25000 [====>.........................] - ETA: 43s - loss: 7.6990 - accuracy: 0.4979
 4768/25000 [====>.........................] - ETA: 43s - loss: 7.6827 - accuracy: 0.4990
 4800/25000 [====>.........................] - ETA: 43s - loss: 7.6890 - accuracy: 0.4985
 4832/25000 [====>.........................] - ETA: 42s - loss: 7.6888 - accuracy: 0.4986
 4864/25000 [====>.........................] - ETA: 42s - loss: 7.6698 - accuracy: 0.4998
 4896/25000 [====>.........................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 4928/25000 [====>.........................] - ETA: 42s - loss: 7.6760 - accuracy: 0.4994
 4960/25000 [====>.........................] - ETA: 42s - loss: 7.6759 - accuracy: 0.4994
 4992/25000 [====>.........................] - ETA: 42s - loss: 7.6850 - accuracy: 0.4988
 5024/25000 [=====>........................] - ETA: 42s - loss: 7.6727 - accuracy: 0.4996
 5056/25000 [=====>........................] - ETA: 42s - loss: 7.6788 - accuracy: 0.4992
 5088/25000 [=====>........................] - ETA: 42s - loss: 7.6757 - accuracy: 0.4994
 5120/25000 [=====>........................] - ETA: 42s - loss: 7.6756 - accuracy: 0.4994
 5152/25000 [=====>........................] - ETA: 42s - loss: 7.6755 - accuracy: 0.4994
 5184/25000 [=====>........................] - ETA: 42s - loss: 7.6725 - accuracy: 0.4996
 5216/25000 [=====>........................] - ETA: 42s - loss: 7.6754 - accuracy: 0.4994
 5248/25000 [=====>........................] - ETA: 41s - loss: 7.6725 - accuracy: 0.4996
 5280/25000 [=====>........................] - ETA: 41s - loss: 7.6782 - accuracy: 0.4992
 5312/25000 [=====>........................] - ETA: 41s - loss: 7.6868 - accuracy: 0.4987
 5344/25000 [=====>........................] - ETA: 41s - loss: 7.6724 - accuracy: 0.4996
 5376/25000 [=====>........................] - ETA: 41s - loss: 7.6780 - accuracy: 0.4993
 5408/25000 [=====>........................] - ETA: 41s - loss: 7.6836 - accuracy: 0.4989
 5440/25000 [=====>........................] - ETA: 41s - loss: 7.6835 - accuracy: 0.4989
 5472/25000 [=====>........................] - ETA: 41s - loss: 7.6750 - accuracy: 0.4995
 5504/25000 [=====>........................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
 5536/25000 [=====>........................] - ETA: 41s - loss: 7.6638 - accuracy: 0.5002
 5568/25000 [=====>........................] - ETA: 41s - loss: 7.6749 - accuracy: 0.4995
 5600/25000 [=====>........................] - ETA: 41s - loss: 7.6776 - accuracy: 0.4993
 5632/25000 [=====>........................] - ETA: 41s - loss: 7.6693 - accuracy: 0.4998
 5664/25000 [=====>........................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
 5696/25000 [=====>........................] - ETA: 41s - loss: 7.6747 - accuracy: 0.4995
 5728/25000 [=====>........................] - ETA: 40s - loss: 7.6773 - accuracy: 0.4993
 5760/25000 [=====>........................] - ETA: 40s - loss: 7.6773 - accuracy: 0.4993
 5792/25000 [=====>........................] - ETA: 40s - loss: 7.6772 - accuracy: 0.4993
 5824/25000 [=====>........................] - ETA: 40s - loss: 7.6587 - accuracy: 0.5005
 5856/25000 [======>.......................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 5888/25000 [======>.......................] - ETA: 40s - loss: 7.6640 - accuracy: 0.5002
 5920/25000 [======>.......................] - ETA: 40s - loss: 7.6692 - accuracy: 0.4998
 5952/25000 [======>.......................] - ETA: 40s - loss: 7.6795 - accuracy: 0.4992
 5984/25000 [======>.......................] - ETA: 40s - loss: 7.6871 - accuracy: 0.4987
 6016/25000 [======>.......................] - ETA: 40s - loss: 7.6819 - accuracy: 0.4990
 6048/25000 [======>.......................] - ETA: 40s - loss: 7.6818 - accuracy: 0.4990
 6080/25000 [======>.......................] - ETA: 40s - loss: 7.6893 - accuracy: 0.4985
 6112/25000 [======>.......................] - ETA: 40s - loss: 7.6842 - accuracy: 0.4989
 6144/25000 [======>.......................] - ETA: 40s - loss: 7.6716 - accuracy: 0.4997
 6176/25000 [======>.......................] - ETA: 40s - loss: 7.6865 - accuracy: 0.4987
 6208/25000 [======>.......................] - ETA: 40s - loss: 7.6864 - accuracy: 0.4987
 6240/25000 [======>.......................] - ETA: 39s - loss: 7.6789 - accuracy: 0.4992
 6272/25000 [======>.......................] - ETA: 39s - loss: 7.6691 - accuracy: 0.4998
 6304/25000 [======>.......................] - ETA: 39s - loss: 7.6739 - accuracy: 0.4995
 6336/25000 [======>.......................] - ETA: 39s - loss: 7.6715 - accuracy: 0.4997
 6368/25000 [======>.......................] - ETA: 39s - loss: 7.6811 - accuracy: 0.4991
 6400/25000 [======>.......................] - ETA: 39s - loss: 7.6906 - accuracy: 0.4984
 6432/25000 [======>.......................] - ETA: 39s - loss: 7.6976 - accuracy: 0.4980
 6464/25000 [======>.......................] - ETA: 39s - loss: 7.7022 - accuracy: 0.4977
 6496/25000 [======>.......................] - ETA: 39s - loss: 7.7067 - accuracy: 0.4974
 6528/25000 [======>.......................] - ETA: 39s - loss: 7.6995 - accuracy: 0.4979
 6560/25000 [======>.......................] - ETA: 39s - loss: 7.6993 - accuracy: 0.4979
 6592/25000 [======>.......................] - ETA: 39s - loss: 7.7015 - accuracy: 0.4977
 6624/25000 [======>.......................] - ETA: 39s - loss: 7.6944 - accuracy: 0.4982
 6656/25000 [======>.......................] - ETA: 38s - loss: 7.6966 - accuracy: 0.4980
 6688/25000 [=======>......................] - ETA: 38s - loss: 7.6987 - accuracy: 0.4979
 6720/25000 [=======>......................] - ETA: 38s - loss: 7.7054 - accuracy: 0.4975
 6752/25000 [=======>......................] - ETA: 38s - loss: 7.6939 - accuracy: 0.4982
 6784/25000 [=======>......................] - ETA: 38s - loss: 7.6983 - accuracy: 0.4979
 6816/25000 [=======>......................] - ETA: 38s - loss: 7.7004 - accuracy: 0.4978
 6848/25000 [=======>......................] - ETA: 38s - loss: 7.6935 - accuracy: 0.4982
 6880/25000 [=======>......................] - ETA: 38s - loss: 7.7000 - accuracy: 0.4978
 6912/25000 [=======>......................] - ETA: 38s - loss: 7.6977 - accuracy: 0.4980
 6944/25000 [=======>......................] - ETA: 38s - loss: 7.7042 - accuracy: 0.4976
 6976/25000 [=======>......................] - ETA: 38s - loss: 7.7084 - accuracy: 0.4973
 7008/25000 [=======>......................] - ETA: 38s - loss: 7.7169 - accuracy: 0.4967
 7040/25000 [=======>......................] - ETA: 38s - loss: 7.7080 - accuracy: 0.4973
 7072/25000 [=======>......................] - ETA: 38s - loss: 7.7165 - accuracy: 0.4967
 7104/25000 [=======>......................] - ETA: 37s - loss: 7.7184 - accuracy: 0.4966
 7136/25000 [=======>......................] - ETA: 37s - loss: 7.7160 - accuracy: 0.4968
 7168/25000 [=======>......................] - ETA: 37s - loss: 7.7158 - accuracy: 0.4968
 7200/25000 [=======>......................] - ETA: 37s - loss: 7.7007 - accuracy: 0.4978
 7232/25000 [=======>......................] - ETA: 37s - loss: 7.7133 - accuracy: 0.4970
 7264/25000 [=======>......................] - ETA: 37s - loss: 7.7109 - accuracy: 0.4971
 7296/25000 [=======>......................] - ETA: 37s - loss: 7.7108 - accuracy: 0.4971
 7328/25000 [=======>......................] - ETA: 37s - loss: 7.7085 - accuracy: 0.4973
 7360/25000 [=======>......................] - ETA: 37s - loss: 7.7104 - accuracy: 0.4971
 7392/25000 [=======>......................] - ETA: 37s - loss: 7.6957 - accuracy: 0.4981
 7424/25000 [=======>......................] - ETA: 37s - loss: 7.7017 - accuracy: 0.4977
 7456/25000 [=======>......................] - ETA: 37s - loss: 7.6913 - accuracy: 0.4984
 7488/25000 [=======>......................] - ETA: 37s - loss: 7.7014 - accuracy: 0.4977
 7520/25000 [========>.....................] - ETA: 37s - loss: 7.6952 - accuracy: 0.4981
 7552/25000 [========>.....................] - ETA: 36s - loss: 7.6991 - accuracy: 0.4979
 7584/25000 [========>.....................] - ETA: 36s - loss: 7.6929 - accuracy: 0.4983
 7616/25000 [========>.....................] - ETA: 36s - loss: 7.6948 - accuracy: 0.4982
 7648/25000 [========>.....................] - ETA: 36s - loss: 7.6967 - accuracy: 0.4980
 7680/25000 [========>.....................] - ETA: 36s - loss: 7.6986 - accuracy: 0.4979
 7712/25000 [========>.....................] - ETA: 36s - loss: 7.7044 - accuracy: 0.4975
 7744/25000 [========>.....................] - ETA: 36s - loss: 7.7062 - accuracy: 0.4974
 7776/25000 [========>.....................] - ETA: 36s - loss: 7.6982 - accuracy: 0.4979
 7808/25000 [========>.....................] - ETA: 36s - loss: 7.7157 - accuracy: 0.4968
 7840/25000 [========>.....................] - ETA: 36s - loss: 7.7233 - accuracy: 0.4963
 7872/25000 [========>.....................] - ETA: 36s - loss: 7.7309 - accuracy: 0.4958
 7904/25000 [========>.....................] - ETA: 36s - loss: 7.7326 - accuracy: 0.4957
 7936/25000 [========>.....................] - ETA: 36s - loss: 7.7246 - accuracy: 0.4962
 7968/25000 [========>.....................] - ETA: 36s - loss: 7.7263 - accuracy: 0.4961
 8000/25000 [========>.....................] - ETA: 36s - loss: 7.7203 - accuracy: 0.4965
 8032/25000 [========>.....................] - ETA: 35s - loss: 7.7258 - accuracy: 0.4961
 8064/25000 [========>.....................] - ETA: 35s - loss: 7.7275 - accuracy: 0.4960
 8096/25000 [========>.....................] - ETA: 35s - loss: 7.7310 - accuracy: 0.4958
 8128/25000 [========>.....................] - ETA: 35s - loss: 7.7251 - accuracy: 0.4962
 8160/25000 [========>.....................] - ETA: 35s - loss: 7.7192 - accuracy: 0.4966
 8192/25000 [========>.....................] - ETA: 35s - loss: 7.7190 - accuracy: 0.4966
 8224/25000 [========>.....................] - ETA: 35s - loss: 7.7226 - accuracy: 0.4964
 8256/25000 [========>.....................] - ETA: 35s - loss: 7.7279 - accuracy: 0.4960
 8288/25000 [========>.....................] - ETA: 35s - loss: 7.7221 - accuracy: 0.4964
 8320/25000 [========>.....................] - ETA: 35s - loss: 7.7293 - accuracy: 0.4959
 8352/25000 [=========>....................] - ETA: 35s - loss: 7.7309 - accuracy: 0.4958
 8384/25000 [=========>....................] - ETA: 35s - loss: 7.7288 - accuracy: 0.4959
 8416/25000 [=========>....................] - ETA: 35s - loss: 7.7213 - accuracy: 0.4964
 8448/25000 [=========>....................] - ETA: 34s - loss: 7.7247 - accuracy: 0.4962
 8480/25000 [=========>....................] - ETA: 34s - loss: 7.7281 - accuracy: 0.4960
 8512/25000 [=========>....................] - ETA: 34s - loss: 7.7189 - accuracy: 0.4966
 8544/25000 [=========>....................] - ETA: 34s - loss: 7.7079 - accuracy: 0.4973
 8576/25000 [=========>....................] - ETA: 34s - loss: 7.7149 - accuracy: 0.4969
 8608/25000 [=========>....................] - ETA: 34s - loss: 7.7236 - accuracy: 0.4963
 8640/25000 [=========>....................] - ETA: 34s - loss: 7.7252 - accuracy: 0.4962
 8672/25000 [=========>....................] - ETA: 34s - loss: 7.7179 - accuracy: 0.4967
 8704/25000 [=========>....................] - ETA: 34s - loss: 7.7265 - accuracy: 0.4961
 8736/25000 [=========>....................] - ETA: 34s - loss: 7.7228 - accuracy: 0.4963
 8768/25000 [=========>....................] - ETA: 34s - loss: 7.7278 - accuracy: 0.4960
 8800/25000 [=========>....................] - ETA: 34s - loss: 7.7415 - accuracy: 0.4951
 8832/25000 [=========>....................] - ETA: 34s - loss: 7.7447 - accuracy: 0.4949
 8864/25000 [=========>....................] - ETA: 34s - loss: 7.7410 - accuracy: 0.4951
 8896/25000 [=========>....................] - ETA: 34s - loss: 7.7304 - accuracy: 0.4958
 8928/25000 [=========>....................] - ETA: 33s - loss: 7.7353 - accuracy: 0.4955
 8960/25000 [=========>....................] - ETA: 33s - loss: 7.7351 - accuracy: 0.4955
 8992/25000 [=========>....................] - ETA: 33s - loss: 7.7297 - accuracy: 0.4959
 9024/25000 [=========>....................] - ETA: 33s - loss: 7.7363 - accuracy: 0.4955
 9056/25000 [=========>....................] - ETA: 33s - loss: 7.7360 - accuracy: 0.4955
 9088/25000 [=========>....................] - ETA: 33s - loss: 7.7392 - accuracy: 0.4953
 9120/25000 [=========>....................] - ETA: 33s - loss: 7.7473 - accuracy: 0.4947
 9152/25000 [=========>....................] - ETA: 33s - loss: 7.7454 - accuracy: 0.4949
 9184/25000 [==========>...................] - ETA: 33s - loss: 7.7501 - accuracy: 0.4946
 9216/25000 [==========>...................] - ETA: 33s - loss: 7.7481 - accuracy: 0.4947
 9248/25000 [==========>...................] - ETA: 33s - loss: 7.7545 - accuracy: 0.4943
 9280/25000 [==========>...................] - ETA: 33s - loss: 7.7509 - accuracy: 0.4945
 9312/25000 [==========>...................] - ETA: 33s - loss: 7.7490 - accuracy: 0.4946
 9344/25000 [==========>...................] - ETA: 33s - loss: 7.7454 - accuracy: 0.4949
 9376/25000 [==========>...................] - ETA: 32s - loss: 7.7533 - accuracy: 0.4943
 9408/25000 [==========>...................] - ETA: 32s - loss: 7.7546 - accuracy: 0.4943
 9440/25000 [==========>...................] - ETA: 32s - loss: 7.7608 - accuracy: 0.4939
 9472/25000 [==========>...................] - ETA: 32s - loss: 7.7637 - accuracy: 0.4937
 9504/25000 [==========>...................] - ETA: 32s - loss: 7.7699 - accuracy: 0.4933
 9536/25000 [==========>...................] - ETA: 32s - loss: 7.7647 - accuracy: 0.4936
 9568/25000 [==========>...................] - ETA: 32s - loss: 7.7628 - accuracy: 0.4937
 9600/25000 [==========>...................] - ETA: 32s - loss: 7.7609 - accuracy: 0.4939
 9632/25000 [==========>...................] - ETA: 32s - loss: 7.7605 - accuracy: 0.4939
 9664/25000 [==========>...................] - ETA: 32s - loss: 7.7650 - accuracy: 0.4936
 9696/25000 [==========>...................] - ETA: 32s - loss: 7.7694 - accuracy: 0.4933
 9728/25000 [==========>...................] - ETA: 32s - loss: 7.7722 - accuracy: 0.4931
 9760/25000 [==========>...................] - ETA: 32s - loss: 7.7703 - accuracy: 0.4932
 9792/25000 [==========>...................] - ETA: 32s - loss: 7.7715 - accuracy: 0.4932
 9824/25000 [==========>...................] - ETA: 32s - loss: 7.7665 - accuracy: 0.4935
 9856/25000 [==========>...................] - ETA: 31s - loss: 7.7584 - accuracy: 0.4940
 9888/25000 [==========>...................] - ETA: 31s - loss: 7.7566 - accuracy: 0.4941
 9920/25000 [==========>...................] - ETA: 31s - loss: 7.7625 - accuracy: 0.4938
 9952/25000 [==========>...................] - ETA: 31s - loss: 7.7606 - accuracy: 0.4939
 9984/25000 [==========>...................] - ETA: 31s - loss: 7.7664 - accuracy: 0.4935
10016/25000 [===========>..................] - ETA: 31s - loss: 7.7615 - accuracy: 0.4938
10048/25000 [===========>..................] - ETA: 31s - loss: 7.7628 - accuracy: 0.4937
10080/25000 [===========>..................] - ETA: 31s - loss: 7.7625 - accuracy: 0.4938
10112/25000 [===========>..................] - ETA: 31s - loss: 7.7728 - accuracy: 0.4931
10144/25000 [===========>..................] - ETA: 31s - loss: 7.7739 - accuracy: 0.4930
10176/25000 [===========>..................] - ETA: 31s - loss: 7.7676 - accuracy: 0.4934
10208/25000 [===========>..................] - ETA: 31s - loss: 7.7778 - accuracy: 0.4928
10240/25000 [===========>..................] - ETA: 31s - loss: 7.7789 - accuracy: 0.4927
10272/25000 [===========>..................] - ETA: 31s - loss: 7.7786 - accuracy: 0.4927
10304/25000 [===========>..................] - ETA: 30s - loss: 7.7812 - accuracy: 0.4925
10336/25000 [===========>..................] - ETA: 30s - loss: 7.7823 - accuracy: 0.4925
10368/25000 [===========>..................] - ETA: 30s - loss: 7.7775 - accuracy: 0.4928
10400/25000 [===========>..................] - ETA: 30s - loss: 7.7742 - accuracy: 0.4930
10432/25000 [===========>..................] - ETA: 30s - loss: 7.7710 - accuracy: 0.4932
10464/25000 [===========>..................] - ETA: 30s - loss: 7.7707 - accuracy: 0.4932
10496/25000 [===========>..................] - ETA: 30s - loss: 7.7718 - accuracy: 0.4931
10528/25000 [===========>..................] - ETA: 30s - loss: 7.7700 - accuracy: 0.4933
10560/25000 [===========>..................] - ETA: 30s - loss: 7.7726 - accuracy: 0.4931
10592/25000 [===========>..................] - ETA: 30s - loss: 7.7766 - accuracy: 0.4928
10624/25000 [===========>..................] - ETA: 30s - loss: 7.7749 - accuracy: 0.4929
10656/25000 [===========>..................] - ETA: 30s - loss: 7.7760 - accuracy: 0.4929
10688/25000 [===========>..................] - ETA: 30s - loss: 7.7728 - accuracy: 0.4931
10720/25000 [===========>..................] - ETA: 30s - loss: 7.7667 - accuracy: 0.4935
10752/25000 [===========>..................] - ETA: 30s - loss: 7.7693 - accuracy: 0.4933
10784/25000 [===========>..................] - ETA: 29s - loss: 7.7747 - accuracy: 0.4930
10816/25000 [===========>..................] - ETA: 29s - loss: 7.7758 - accuracy: 0.4929
10848/25000 [============>.................] - ETA: 29s - loss: 7.7712 - accuracy: 0.4932
10880/25000 [============>.................] - ETA: 29s - loss: 7.7723 - accuracy: 0.4931
10912/25000 [============>.................] - ETA: 29s - loss: 7.7720 - accuracy: 0.4931
10944/25000 [============>.................] - ETA: 29s - loss: 7.7731 - accuracy: 0.4931
10976/25000 [============>.................] - ETA: 29s - loss: 7.7728 - accuracy: 0.4931
11008/25000 [============>.................] - ETA: 29s - loss: 7.7683 - accuracy: 0.4934
11040/25000 [============>.................] - ETA: 29s - loss: 7.7625 - accuracy: 0.4938
11072/25000 [============>.................] - ETA: 29s - loss: 7.7622 - accuracy: 0.4938
11104/25000 [============>.................] - ETA: 29s - loss: 7.7619 - accuracy: 0.4938
11136/25000 [============>.................] - ETA: 29s - loss: 7.7575 - accuracy: 0.4941
11168/25000 [============>.................] - ETA: 29s - loss: 7.7641 - accuracy: 0.4936
11200/25000 [============>.................] - ETA: 29s - loss: 7.7597 - accuracy: 0.4939
11232/25000 [============>.................] - ETA: 29s - loss: 7.7540 - accuracy: 0.4943
11264/25000 [============>.................] - ETA: 28s - loss: 7.7524 - accuracy: 0.4944
11296/25000 [============>.................] - ETA: 28s - loss: 7.7453 - accuracy: 0.4949
11328/25000 [============>.................] - ETA: 28s - loss: 7.7438 - accuracy: 0.4950
11360/25000 [============>.................] - ETA: 28s - loss: 7.7463 - accuracy: 0.4948
11392/25000 [============>.................] - ETA: 28s - loss: 7.7433 - accuracy: 0.4950
11424/25000 [============>.................] - ETA: 28s - loss: 7.7431 - accuracy: 0.4950
11456/25000 [============>.................] - ETA: 28s - loss: 7.7376 - accuracy: 0.4954
11488/25000 [============>.................] - ETA: 28s - loss: 7.7360 - accuracy: 0.4955
11520/25000 [============>.................] - ETA: 28s - loss: 7.7412 - accuracy: 0.4951
11552/25000 [============>.................] - ETA: 28s - loss: 7.7409 - accuracy: 0.4952
11584/25000 [============>.................] - ETA: 28s - loss: 7.7394 - accuracy: 0.4953
11616/25000 [============>.................] - ETA: 28s - loss: 7.7445 - accuracy: 0.4949
11648/25000 [============>.................] - ETA: 28s - loss: 7.7469 - accuracy: 0.4948
11680/25000 [=============>................] - ETA: 28s - loss: 7.7572 - accuracy: 0.4941
11712/25000 [=============>................] - ETA: 27s - loss: 7.7530 - accuracy: 0.4944
11744/25000 [=============>................] - ETA: 27s - loss: 7.7437 - accuracy: 0.4950
11776/25000 [=============>................] - ETA: 27s - loss: 7.7473 - accuracy: 0.4947
11808/25000 [=============>................] - ETA: 27s - loss: 7.7458 - accuracy: 0.4948
11840/25000 [=============>................] - ETA: 27s - loss: 7.7430 - accuracy: 0.4950
11872/25000 [=============>................] - ETA: 27s - loss: 7.7454 - accuracy: 0.4949
11904/25000 [=============>................] - ETA: 27s - loss: 7.7465 - accuracy: 0.4948
11936/25000 [=============>................] - ETA: 27s - loss: 7.7488 - accuracy: 0.4946
11968/25000 [=============>................] - ETA: 27s - loss: 7.7512 - accuracy: 0.4945
12000/25000 [=============>................] - ETA: 27s - loss: 7.7497 - accuracy: 0.4946
12032/25000 [=============>................] - ETA: 27s - loss: 7.7469 - accuracy: 0.4948
12064/25000 [=============>................] - ETA: 27s - loss: 7.7530 - accuracy: 0.4944
12096/25000 [=============>................] - ETA: 27s - loss: 7.7528 - accuracy: 0.4944
12128/25000 [=============>................] - ETA: 27s - loss: 7.7551 - accuracy: 0.4942
12160/25000 [=============>................] - ETA: 27s - loss: 7.7587 - accuracy: 0.4940
12192/25000 [=============>................] - ETA: 26s - loss: 7.7597 - accuracy: 0.4939
12224/25000 [=============>................] - ETA: 26s - loss: 7.7670 - accuracy: 0.4935
12256/25000 [=============>................] - ETA: 26s - loss: 7.7655 - accuracy: 0.4936
12288/25000 [=============>................] - ETA: 26s - loss: 7.7664 - accuracy: 0.4935
12320/25000 [=============>................] - ETA: 26s - loss: 7.7687 - accuracy: 0.4933
12352/25000 [=============>................] - ETA: 26s - loss: 7.7672 - accuracy: 0.4934
12384/25000 [=============>................] - ETA: 26s - loss: 7.7743 - accuracy: 0.4930
12416/25000 [=============>................] - ETA: 26s - loss: 7.7753 - accuracy: 0.4929
12448/25000 [=============>................] - ETA: 26s - loss: 7.7738 - accuracy: 0.4930
12480/25000 [=============>................] - ETA: 26s - loss: 7.7735 - accuracy: 0.4930
12512/25000 [==============>...............] - ETA: 26s - loss: 7.7708 - accuracy: 0.4932
12544/25000 [==============>...............] - ETA: 26s - loss: 7.7717 - accuracy: 0.4931
12576/25000 [==============>...............] - ETA: 26s - loss: 7.7727 - accuracy: 0.4931
12608/25000 [==============>...............] - ETA: 26s - loss: 7.7724 - accuracy: 0.4931
12640/25000 [==============>...............] - ETA: 25s - loss: 7.7722 - accuracy: 0.4931
12672/25000 [==============>...............] - ETA: 25s - loss: 7.7731 - accuracy: 0.4931
12704/25000 [==============>...............] - ETA: 25s - loss: 7.7716 - accuracy: 0.4932
12736/25000 [==============>...............] - ETA: 25s - loss: 7.7690 - accuracy: 0.4933
12768/25000 [==============>...............] - ETA: 25s - loss: 7.7723 - accuracy: 0.4931
12800/25000 [==============>...............] - ETA: 25s - loss: 7.7684 - accuracy: 0.4934
12832/25000 [==============>...............] - ETA: 25s - loss: 7.7706 - accuracy: 0.4932
12864/25000 [==============>...............] - ETA: 25s - loss: 7.7727 - accuracy: 0.4931
12896/25000 [==============>...............] - ETA: 25s - loss: 7.7760 - accuracy: 0.4929
12928/25000 [==============>...............] - ETA: 25s - loss: 7.7722 - accuracy: 0.4931
12960/25000 [==============>...............] - ETA: 25s - loss: 7.7731 - accuracy: 0.4931
12992/25000 [==============>...............] - ETA: 25s - loss: 7.7728 - accuracy: 0.4931
13024/25000 [==============>...............] - ETA: 25s - loss: 7.7738 - accuracy: 0.4930
13056/25000 [==============>...............] - ETA: 25s - loss: 7.7747 - accuracy: 0.4930
13088/25000 [==============>...............] - ETA: 25s - loss: 7.7838 - accuracy: 0.4924
13120/25000 [==============>...............] - ETA: 24s - loss: 7.7882 - accuracy: 0.4921
13152/25000 [==============>...............] - ETA: 24s - loss: 7.7925 - accuracy: 0.4918
13184/25000 [==============>...............] - ETA: 24s - loss: 7.7957 - accuracy: 0.4916
13216/25000 [==============>...............] - ETA: 24s - loss: 7.7931 - accuracy: 0.4918
13248/25000 [==============>...............] - ETA: 24s - loss: 7.7905 - accuracy: 0.4919
13280/25000 [==============>...............] - ETA: 24s - loss: 7.7890 - accuracy: 0.4920
13312/25000 [==============>...............] - ETA: 24s - loss: 7.7899 - accuracy: 0.4920
13344/25000 [===============>..............] - ETA: 24s - loss: 7.7907 - accuracy: 0.4919
13376/25000 [===============>..............] - ETA: 24s - loss: 7.7927 - accuracy: 0.4918
13408/25000 [===============>..............] - ETA: 24s - loss: 7.7936 - accuracy: 0.4917
13440/25000 [===============>..............] - ETA: 24s - loss: 7.7921 - accuracy: 0.4918
13472/25000 [===============>..............] - ETA: 24s - loss: 7.7941 - accuracy: 0.4917
13504/25000 [===============>..............] - ETA: 24s - loss: 7.7938 - accuracy: 0.4917
13536/25000 [===============>..............] - ETA: 24s - loss: 7.7958 - accuracy: 0.4916
13568/25000 [===============>..............] - ETA: 23s - loss: 7.7966 - accuracy: 0.4915
13600/25000 [===============>..............] - ETA: 23s - loss: 7.7951 - accuracy: 0.4916
13632/25000 [===============>..............] - ETA: 23s - loss: 7.7926 - accuracy: 0.4918
13664/25000 [===============>..............] - ETA: 23s - loss: 7.7867 - accuracy: 0.4922
13696/25000 [===============>..............] - ETA: 23s - loss: 7.7831 - accuracy: 0.4924
13728/25000 [===============>..............] - ETA: 23s - loss: 7.7828 - accuracy: 0.4924
13760/25000 [===============>..............] - ETA: 23s - loss: 7.7769 - accuracy: 0.4928
13792/25000 [===============>..............] - ETA: 23s - loss: 7.7700 - accuracy: 0.4933
13824/25000 [===============>..............] - ETA: 23s - loss: 7.7698 - accuracy: 0.4933
13856/25000 [===============>..............] - ETA: 23s - loss: 7.7751 - accuracy: 0.4929
13888/25000 [===============>..............] - ETA: 23s - loss: 7.7737 - accuracy: 0.4930
13920/25000 [===============>..............] - ETA: 23s - loss: 7.7702 - accuracy: 0.4932
13952/25000 [===============>..............] - ETA: 23s - loss: 7.7688 - accuracy: 0.4933
13984/25000 [===============>..............] - ETA: 23s - loss: 7.7620 - accuracy: 0.4938
14016/25000 [===============>..............] - ETA: 23s - loss: 7.7596 - accuracy: 0.4939
14048/25000 [===============>..............] - ETA: 22s - loss: 7.7605 - accuracy: 0.4939
14080/25000 [===============>..............] - ETA: 22s - loss: 7.7625 - accuracy: 0.4938
14112/25000 [===============>..............] - ETA: 22s - loss: 7.7633 - accuracy: 0.4937
14144/25000 [===============>..............] - ETA: 22s - loss: 7.7674 - accuracy: 0.4934
14176/25000 [================>.............] - ETA: 22s - loss: 7.7618 - accuracy: 0.4938
14208/25000 [================>.............] - ETA: 22s - loss: 7.7627 - accuracy: 0.4937
14240/25000 [================>.............] - ETA: 22s - loss: 7.7592 - accuracy: 0.4940
14272/25000 [================>.............] - ETA: 22s - loss: 7.7644 - accuracy: 0.4936
14304/25000 [================>.............] - ETA: 22s - loss: 7.7642 - accuracy: 0.4936
14336/25000 [================>.............] - ETA: 22s - loss: 7.7650 - accuracy: 0.4936
14368/25000 [================>.............] - ETA: 22s - loss: 7.7637 - accuracy: 0.4937
14400/25000 [================>.............] - ETA: 22s - loss: 7.7614 - accuracy: 0.4938
14432/25000 [================>.............] - ETA: 22s - loss: 7.7676 - accuracy: 0.4934
14464/25000 [================>.............] - ETA: 22s - loss: 7.7663 - accuracy: 0.4935
14496/25000 [================>.............] - ETA: 21s - loss: 7.7660 - accuracy: 0.4935
14528/25000 [================>.............] - ETA: 21s - loss: 7.7627 - accuracy: 0.4937
14560/25000 [================>.............] - ETA: 21s - loss: 7.7614 - accuracy: 0.4938
14592/25000 [================>.............] - ETA: 21s - loss: 7.7643 - accuracy: 0.4936
14624/25000 [================>.............] - ETA: 21s - loss: 7.7631 - accuracy: 0.4937
14656/25000 [================>.............] - ETA: 21s - loss: 7.7650 - accuracy: 0.4936
14688/25000 [================>.............] - ETA: 21s - loss: 7.7689 - accuracy: 0.4933
14720/25000 [================>.............] - ETA: 21s - loss: 7.7718 - accuracy: 0.4931
14752/25000 [================>.............] - ETA: 21s - loss: 7.7726 - accuracy: 0.4931
14784/25000 [================>.............] - ETA: 21s - loss: 7.7755 - accuracy: 0.4929
14816/25000 [================>.............] - ETA: 21s - loss: 7.7722 - accuracy: 0.4931
14848/25000 [================>.............] - ETA: 21s - loss: 7.7709 - accuracy: 0.4932
14880/25000 [================>.............] - ETA: 21s - loss: 7.7707 - accuracy: 0.4932
14912/25000 [================>.............] - ETA: 21s - loss: 7.7674 - accuracy: 0.4934
14944/25000 [================>.............] - ETA: 21s - loss: 7.7661 - accuracy: 0.4935
14976/25000 [================>.............] - ETA: 20s - loss: 7.7670 - accuracy: 0.4935
15008/25000 [=================>............] - ETA: 20s - loss: 7.7657 - accuracy: 0.4935
15040/25000 [=================>............] - ETA: 20s - loss: 7.7696 - accuracy: 0.4933
15072/25000 [=================>............] - ETA: 20s - loss: 7.7633 - accuracy: 0.4937
15104/25000 [=================>............] - ETA: 20s - loss: 7.7620 - accuracy: 0.4938
15136/25000 [=================>............] - ETA: 20s - loss: 7.7629 - accuracy: 0.4937
15168/25000 [=================>............] - ETA: 20s - loss: 7.7637 - accuracy: 0.4937
15200/25000 [=================>............] - ETA: 20s - loss: 7.7645 - accuracy: 0.4936
15232/25000 [=================>............] - ETA: 20s - loss: 7.7623 - accuracy: 0.4938
15264/25000 [=================>............] - ETA: 20s - loss: 7.7610 - accuracy: 0.4938
15296/25000 [=================>............] - ETA: 20s - loss: 7.7608 - accuracy: 0.4939
15328/25000 [=================>............] - ETA: 20s - loss: 7.7617 - accuracy: 0.4938
15360/25000 [=================>............] - ETA: 20s - loss: 7.7615 - accuracy: 0.4938
15392/25000 [=================>............] - ETA: 20s - loss: 7.7593 - accuracy: 0.4940
15424/25000 [=================>............] - ETA: 20s - loss: 7.7591 - accuracy: 0.4940
15456/25000 [=================>............] - ETA: 19s - loss: 7.7628 - accuracy: 0.4937
15488/25000 [=================>............] - ETA: 19s - loss: 7.7577 - accuracy: 0.4941
15520/25000 [=================>............] - ETA: 19s - loss: 7.7575 - accuracy: 0.4941
15552/25000 [=================>............] - ETA: 19s - loss: 7.7573 - accuracy: 0.4941
15584/25000 [=================>............] - ETA: 19s - loss: 7.7522 - accuracy: 0.4944
15616/25000 [=================>............] - ETA: 19s - loss: 7.7511 - accuracy: 0.4945
15648/25000 [=================>............] - ETA: 19s - loss: 7.7509 - accuracy: 0.4945
15680/25000 [=================>............] - ETA: 19s - loss: 7.7546 - accuracy: 0.4943
15712/25000 [=================>............] - ETA: 19s - loss: 7.7593 - accuracy: 0.4940
15744/25000 [=================>............] - ETA: 19s - loss: 7.7601 - accuracy: 0.4939
15776/25000 [=================>............] - ETA: 19s - loss: 7.7609 - accuracy: 0.4939
15808/25000 [=================>............] - ETA: 19s - loss: 7.7607 - accuracy: 0.4939
15840/25000 [==================>...........] - ETA: 19s - loss: 7.7586 - accuracy: 0.4940
15872/25000 [==================>...........] - ETA: 19s - loss: 7.7594 - accuracy: 0.4940
15904/25000 [==================>...........] - ETA: 18s - loss: 7.7592 - accuracy: 0.4940
15936/25000 [==================>...........] - ETA: 18s - loss: 7.7571 - accuracy: 0.4941
15968/25000 [==================>...........] - ETA: 18s - loss: 7.7550 - accuracy: 0.4942
16000/25000 [==================>...........] - ETA: 18s - loss: 7.7548 - accuracy: 0.4942
16032/25000 [==================>...........] - ETA: 18s - loss: 7.7565 - accuracy: 0.4941
16064/25000 [==================>...........] - ETA: 18s - loss: 7.7516 - accuracy: 0.4945
16096/25000 [==================>...........] - ETA: 18s - loss: 7.7533 - accuracy: 0.4943
16128/25000 [==================>...........] - ETA: 18s - loss: 7.7531 - accuracy: 0.4944
16160/25000 [==================>...........] - ETA: 18s - loss: 7.7492 - accuracy: 0.4946
16192/25000 [==================>...........] - ETA: 18s - loss: 7.7518 - accuracy: 0.4944
16224/25000 [==================>...........] - ETA: 18s - loss: 7.7507 - accuracy: 0.4945
16256/25000 [==================>...........] - ETA: 18s - loss: 7.7477 - accuracy: 0.4947
16288/25000 [==================>...........] - ETA: 18s - loss: 7.7466 - accuracy: 0.4948
16320/25000 [==================>...........] - ETA: 18s - loss: 7.7465 - accuracy: 0.4948
16352/25000 [==================>...........] - ETA: 18s - loss: 7.7435 - accuracy: 0.4950
16384/25000 [==================>...........] - ETA: 17s - loss: 7.7415 - accuracy: 0.4951
16416/25000 [==================>...........] - ETA: 17s - loss: 7.7451 - accuracy: 0.4949
16448/25000 [==================>...........] - ETA: 17s - loss: 7.7449 - accuracy: 0.4949
16480/25000 [==================>...........] - ETA: 17s - loss: 7.7420 - accuracy: 0.4951
16512/25000 [==================>...........] - ETA: 17s - loss: 7.7456 - accuracy: 0.4949
16544/25000 [==================>...........] - ETA: 17s - loss: 7.7426 - accuracy: 0.4950
16576/25000 [==================>...........] - ETA: 17s - loss: 7.7388 - accuracy: 0.4953
16608/25000 [==================>...........] - ETA: 17s - loss: 7.7414 - accuracy: 0.4951
16640/25000 [==================>...........] - ETA: 17s - loss: 7.7376 - accuracy: 0.4954
16672/25000 [===================>..........] - ETA: 17s - loss: 7.7393 - accuracy: 0.4953
16704/25000 [===================>..........] - ETA: 17s - loss: 7.7401 - accuracy: 0.4952
16736/25000 [===================>..........] - ETA: 17s - loss: 7.7381 - accuracy: 0.4953
16768/25000 [===================>..........] - ETA: 17s - loss: 7.7352 - accuracy: 0.4955
16800/25000 [===================>..........] - ETA: 17s - loss: 7.7369 - accuracy: 0.4954
16832/25000 [===================>..........] - ETA: 17s - loss: 7.7359 - accuracy: 0.4955
16864/25000 [===================>..........] - ETA: 16s - loss: 7.7348 - accuracy: 0.4956
16896/25000 [===================>..........] - ETA: 16s - loss: 7.7292 - accuracy: 0.4959
16928/25000 [===================>..........] - ETA: 16s - loss: 7.7291 - accuracy: 0.4959
16960/25000 [===================>..........] - ETA: 16s - loss: 7.7290 - accuracy: 0.4959
16992/25000 [===================>..........] - ETA: 16s - loss: 7.7289 - accuracy: 0.4959
17024/25000 [===================>..........] - ETA: 16s - loss: 7.7297 - accuracy: 0.4959
17056/25000 [===================>..........] - ETA: 16s - loss: 7.7233 - accuracy: 0.4963
17088/25000 [===================>..........] - ETA: 16s - loss: 7.7214 - accuracy: 0.4964
17120/25000 [===================>..........] - ETA: 16s - loss: 7.7150 - accuracy: 0.4968
17152/25000 [===================>..........] - ETA: 16s - loss: 7.7113 - accuracy: 0.4971
17184/25000 [===================>..........] - ETA: 16s - loss: 7.7121 - accuracy: 0.4970
17216/25000 [===================>..........] - ETA: 16s - loss: 7.7129 - accuracy: 0.4970
17248/25000 [===================>..........] - ETA: 16s - loss: 7.7137 - accuracy: 0.4969
17280/25000 [===================>..........] - ETA: 16s - loss: 7.7101 - accuracy: 0.4972
17312/25000 [===================>..........] - ETA: 15s - loss: 7.7127 - accuracy: 0.4970
17344/25000 [===================>..........] - ETA: 15s - loss: 7.7117 - accuracy: 0.4971
17376/25000 [===================>..........] - ETA: 15s - loss: 7.7081 - accuracy: 0.4973
17408/25000 [===================>..........] - ETA: 15s - loss: 7.7098 - accuracy: 0.4972
17440/25000 [===================>..........] - ETA: 15s - loss: 7.7097 - accuracy: 0.4972
17472/25000 [===================>..........] - ETA: 15s - loss: 7.7079 - accuracy: 0.4973
17504/25000 [====================>.........] - ETA: 15s - loss: 7.7043 - accuracy: 0.4975
17536/25000 [====================>.........] - ETA: 15s - loss: 7.7042 - accuracy: 0.4975
17568/25000 [====================>.........] - ETA: 15s - loss: 7.7033 - accuracy: 0.4976
17600/25000 [====================>.........] - ETA: 15s - loss: 7.7084 - accuracy: 0.4973
17632/25000 [====================>.........] - ETA: 15s - loss: 7.7049 - accuracy: 0.4975
17664/25000 [====================>.........] - ETA: 15s - loss: 7.6987 - accuracy: 0.4979
17696/25000 [====================>.........] - ETA: 15s - loss: 7.6952 - accuracy: 0.4981
17728/25000 [====================>.........] - ETA: 15s - loss: 7.6960 - accuracy: 0.4981
17760/25000 [====================>.........] - ETA: 15s - loss: 7.6968 - accuracy: 0.4980
17792/25000 [====================>.........] - ETA: 14s - loss: 7.6959 - accuracy: 0.4981
17824/25000 [====================>.........] - ETA: 14s - loss: 7.6941 - accuracy: 0.4982
17856/25000 [====================>.........] - ETA: 14s - loss: 7.6975 - accuracy: 0.4980
17888/25000 [====================>.........] - ETA: 14s - loss: 7.6949 - accuracy: 0.4982
17920/25000 [====================>.........] - ETA: 14s - loss: 7.6923 - accuracy: 0.4983
17952/25000 [====================>.........] - ETA: 14s - loss: 7.6931 - accuracy: 0.4983
17984/25000 [====================>.........] - ETA: 14s - loss: 7.6956 - accuracy: 0.4981
18016/25000 [====================>.........] - ETA: 14s - loss: 7.6930 - accuracy: 0.4983
18048/25000 [====================>.........] - ETA: 14s - loss: 7.6862 - accuracy: 0.4987
18080/25000 [====================>.........] - ETA: 14s - loss: 7.6861 - accuracy: 0.4987
18112/25000 [====================>.........] - ETA: 14s - loss: 7.6878 - accuracy: 0.4986
18144/25000 [====================>.........] - ETA: 14s - loss: 7.6869 - accuracy: 0.4987
18176/25000 [====================>.........] - ETA: 14s - loss: 7.6869 - accuracy: 0.4987
18208/25000 [====================>.........] - ETA: 14s - loss: 7.6826 - accuracy: 0.4990
18240/25000 [====================>.........] - ETA: 14s - loss: 7.6809 - accuracy: 0.4991
18272/25000 [====================>.........] - ETA: 13s - loss: 7.6800 - accuracy: 0.4991
18304/25000 [====================>.........] - ETA: 13s - loss: 7.6767 - accuracy: 0.4993
18336/25000 [=====================>........] - ETA: 13s - loss: 7.6758 - accuracy: 0.4994
18368/25000 [=====================>........] - ETA: 13s - loss: 7.6758 - accuracy: 0.4994
18400/25000 [=====================>........] - ETA: 13s - loss: 7.6758 - accuracy: 0.4994
18432/25000 [=====================>........] - ETA: 13s - loss: 7.6758 - accuracy: 0.4994
18464/25000 [=====================>........] - ETA: 13s - loss: 7.6782 - accuracy: 0.4992
18496/25000 [=====================>........] - ETA: 13s - loss: 7.6766 - accuracy: 0.4994
18528/25000 [=====================>........] - ETA: 13s - loss: 7.6766 - accuracy: 0.4994
18560/25000 [=====================>........] - ETA: 13s - loss: 7.6774 - accuracy: 0.4993
18592/25000 [=====================>........] - ETA: 13s - loss: 7.6815 - accuracy: 0.4990
18624/25000 [=====================>........] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
18656/25000 [=====================>........] - ETA: 13s - loss: 7.6855 - accuracy: 0.4988
18688/25000 [=====================>........] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
18720/25000 [=====================>........] - ETA: 13s - loss: 7.6838 - accuracy: 0.4989
18752/25000 [=====================>........] - ETA: 12s - loss: 7.6871 - accuracy: 0.4987
18784/25000 [=====================>........] - ETA: 12s - loss: 7.6895 - accuracy: 0.4985
18816/25000 [=====================>........] - ETA: 12s - loss: 7.6919 - accuracy: 0.4984
18848/25000 [=====================>........] - ETA: 12s - loss: 7.6886 - accuracy: 0.4986
18880/25000 [=====================>........] - ETA: 12s - loss: 7.6877 - accuracy: 0.4986
18912/25000 [=====================>........] - ETA: 12s - loss: 7.6869 - accuracy: 0.4987
18944/25000 [=====================>........] - ETA: 12s - loss: 7.6925 - accuracy: 0.4983
18976/25000 [=====================>........] - ETA: 12s - loss: 7.6917 - accuracy: 0.4984
19008/25000 [=====================>........] - ETA: 12s - loss: 7.6908 - accuracy: 0.4984
19040/25000 [=====================>........] - ETA: 12s - loss: 7.6908 - accuracy: 0.4984
19072/25000 [=====================>........] - ETA: 12s - loss: 7.6899 - accuracy: 0.4985
19104/25000 [=====================>........] - ETA: 12s - loss: 7.6883 - accuracy: 0.4986
19136/25000 [=====================>........] - ETA: 12s - loss: 7.6875 - accuracy: 0.4986
19168/25000 [======================>.......] - ETA: 12s - loss: 7.6858 - accuracy: 0.4987
19200/25000 [======================>.......] - ETA: 12s - loss: 7.6842 - accuracy: 0.4989
19232/25000 [======================>.......] - ETA: 11s - loss: 7.6802 - accuracy: 0.4991
19264/25000 [======================>.......] - ETA: 11s - loss: 7.6809 - accuracy: 0.4991
19296/25000 [======================>.......] - ETA: 11s - loss: 7.6785 - accuracy: 0.4992
19328/25000 [======================>.......] - ETA: 11s - loss: 7.6809 - accuracy: 0.4991
19360/25000 [======================>.......] - ETA: 11s - loss: 7.6801 - accuracy: 0.4991
19392/25000 [======================>.......] - ETA: 11s - loss: 7.6801 - accuracy: 0.4991
19424/25000 [======================>.......] - ETA: 11s - loss: 7.6800 - accuracy: 0.4991
19456/25000 [======================>.......] - ETA: 11s - loss: 7.6792 - accuracy: 0.4992
19488/25000 [======================>.......] - ETA: 11s - loss: 7.6745 - accuracy: 0.4995
19520/25000 [======================>.......] - ETA: 11s - loss: 7.6776 - accuracy: 0.4993
19552/25000 [======================>.......] - ETA: 11s - loss: 7.6760 - accuracy: 0.4994
19584/25000 [======================>.......] - ETA: 11s - loss: 7.6729 - accuracy: 0.4996
19616/25000 [======================>.......] - ETA: 11s - loss: 7.6729 - accuracy: 0.4996
19648/25000 [======================>.......] - ETA: 11s - loss: 7.6736 - accuracy: 0.4995
19680/25000 [======================>.......] - ETA: 11s - loss: 7.6767 - accuracy: 0.4993
19712/25000 [======================>.......] - ETA: 10s - loss: 7.6760 - accuracy: 0.4994
19744/25000 [======================>.......] - ETA: 10s - loss: 7.6736 - accuracy: 0.4995
19776/25000 [======================>.......] - ETA: 10s - loss: 7.6728 - accuracy: 0.4996
19808/25000 [======================>.......] - ETA: 10s - loss: 7.6728 - accuracy: 0.4996
19840/25000 [======================>.......] - ETA: 10s - loss: 7.6705 - accuracy: 0.4997
19872/25000 [======================>.......] - ETA: 10s - loss: 7.6682 - accuracy: 0.4999
19904/25000 [======================>.......] - ETA: 10s - loss: 7.6689 - accuracy: 0.4998
19936/25000 [======================>.......] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
19968/25000 [======================>.......] - ETA: 10s - loss: 7.6689 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 10s - loss: 7.6689 - accuracy: 0.4999
20032/25000 [=======================>......] - ETA: 10s - loss: 7.6682 - accuracy: 0.4999
20064/25000 [=======================>......] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20096/25000 [=======================>......] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
20128/25000 [=======================>......] - ETA: 10s - loss: 7.6643 - accuracy: 0.5001
20160/25000 [=======================>......] - ETA: 10s - loss: 7.6651 - accuracy: 0.5001
20192/25000 [=======================>......] - ETA: 9s - loss: 7.6674 - accuracy: 0.5000 
20224/25000 [=======================>......] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
20256/25000 [=======================>......] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999
20288/25000 [=======================>......] - ETA: 9s - loss: 7.6719 - accuracy: 0.4997
20320/25000 [=======================>......] - ETA: 9s - loss: 7.6742 - accuracy: 0.4995
20352/25000 [=======================>......] - ETA: 9s - loss: 7.6749 - accuracy: 0.4995
20384/25000 [=======================>......] - ETA: 9s - loss: 7.6734 - accuracy: 0.4996
20416/25000 [=======================>......] - ETA: 9s - loss: 7.6696 - accuracy: 0.4998
20448/25000 [=======================>......] - ETA: 9s - loss: 7.6719 - accuracy: 0.4997
20480/25000 [=======================>......] - ETA: 9s - loss: 7.6734 - accuracy: 0.4996
20512/25000 [=======================>......] - ETA: 9s - loss: 7.6748 - accuracy: 0.4995
20544/25000 [=======================>......] - ETA: 9s - loss: 7.6778 - accuracy: 0.4993
20576/25000 [=======================>......] - ETA: 9s - loss: 7.6763 - accuracy: 0.4994
20608/25000 [=======================>......] - ETA: 9s - loss: 7.6778 - accuracy: 0.4993
20640/25000 [=======================>......] - ETA: 9s - loss: 7.6778 - accuracy: 0.4993
20672/25000 [=======================>......] - ETA: 8s - loss: 7.6763 - accuracy: 0.4994
20704/25000 [=======================>......] - ETA: 8s - loss: 7.6785 - accuracy: 0.4992
20736/25000 [=======================>......] - ETA: 8s - loss: 7.6792 - accuracy: 0.4992
20768/25000 [=======================>......] - ETA: 8s - loss: 7.6777 - accuracy: 0.4993
20800/25000 [=======================>......] - ETA: 8s - loss: 7.6799 - accuracy: 0.4991
20832/25000 [=======================>......] - ETA: 8s - loss: 7.6806 - accuracy: 0.4991
20864/25000 [========================>.....] - ETA: 8s - loss: 7.6791 - accuracy: 0.4992
20896/25000 [========================>.....] - ETA: 8s - loss: 7.6806 - accuracy: 0.4991
20928/25000 [========================>.....] - ETA: 8s - loss: 7.6791 - accuracy: 0.4992
20960/25000 [========================>.....] - ETA: 8s - loss: 7.6769 - accuracy: 0.4993
20992/25000 [========================>.....] - ETA: 8s - loss: 7.6805 - accuracy: 0.4991
21024/25000 [========================>.....] - ETA: 8s - loss: 7.6812 - accuracy: 0.4990
21056/25000 [========================>.....] - ETA: 8s - loss: 7.6834 - accuracy: 0.4989
21088/25000 [========================>.....] - ETA: 8s - loss: 7.6819 - accuracy: 0.4990
21120/25000 [========================>.....] - ETA: 8s - loss: 7.6811 - accuracy: 0.4991
21152/25000 [========================>.....] - ETA: 7s - loss: 7.6804 - accuracy: 0.4991
21184/25000 [========================>.....] - ETA: 7s - loss: 7.6775 - accuracy: 0.4993
21216/25000 [========================>.....] - ETA: 7s - loss: 7.6789 - accuracy: 0.4992
21248/25000 [========================>.....] - ETA: 7s - loss: 7.6811 - accuracy: 0.4991
21280/25000 [========================>.....] - ETA: 7s - loss: 7.6803 - accuracy: 0.4991
21312/25000 [========================>.....] - ETA: 7s - loss: 7.6803 - accuracy: 0.4991
21344/25000 [========================>.....] - ETA: 7s - loss: 7.6788 - accuracy: 0.4992
21376/25000 [========================>.....] - ETA: 7s - loss: 7.6795 - accuracy: 0.4992
21408/25000 [========================>.....] - ETA: 7s - loss: 7.6824 - accuracy: 0.4990
21440/25000 [========================>.....] - ETA: 7s - loss: 7.6824 - accuracy: 0.4990
21472/25000 [========================>.....] - ETA: 7s - loss: 7.6838 - accuracy: 0.4989
21504/25000 [========================>.....] - ETA: 7s - loss: 7.6816 - accuracy: 0.4990
21536/25000 [========================>.....] - ETA: 7s - loss: 7.6816 - accuracy: 0.4990
21568/25000 [========================>.....] - ETA: 7s - loss: 7.6801 - accuracy: 0.4991
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6744 - accuracy: 0.4995
21632/25000 [========================>.....] - ETA: 6s - loss: 7.6751 - accuracy: 0.4994
21664/25000 [========================>.....] - ETA: 6s - loss: 7.6744 - accuracy: 0.4995
21696/25000 [=========================>....] - ETA: 6s - loss: 7.6744 - accuracy: 0.4995
21728/25000 [=========================>....] - ETA: 6s - loss: 7.6737 - accuracy: 0.4995
21760/25000 [=========================>....] - ETA: 6s - loss: 7.6723 - accuracy: 0.4996
21792/25000 [=========================>....] - ETA: 6s - loss: 7.6730 - accuracy: 0.4996
21824/25000 [=========================>....] - ETA: 6s - loss: 7.6694 - accuracy: 0.4998
21856/25000 [=========================>....] - ETA: 6s - loss: 7.6659 - accuracy: 0.5000
21888/25000 [=========================>....] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
21920/25000 [=========================>....] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
21952/25000 [=========================>....] - ETA: 6s - loss: 7.6701 - accuracy: 0.4998
21984/25000 [=========================>....] - ETA: 6s - loss: 7.6708 - accuracy: 0.4997
22016/25000 [=========================>....] - ETA: 6s - loss: 7.6715 - accuracy: 0.4997
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6701 - accuracy: 0.4998
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6708 - accuracy: 0.4997
22112/25000 [=========================>....] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22144/25000 [=========================>....] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
22176/25000 [=========================>....] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
22208/25000 [=========================>....] - ETA: 5s - loss: 7.6652 - accuracy: 0.5001
22240/25000 [=========================>....] - ETA: 5s - loss: 7.6618 - accuracy: 0.5003
22272/25000 [=========================>....] - ETA: 5s - loss: 7.6597 - accuracy: 0.5004
22304/25000 [=========================>....] - ETA: 5s - loss: 7.6591 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 5s - loss: 7.6591 - accuracy: 0.5005
22368/25000 [=========================>....] - ETA: 5s - loss: 7.6563 - accuracy: 0.5007
22400/25000 [=========================>....] - ETA: 5s - loss: 7.6543 - accuracy: 0.5008
22432/25000 [=========================>....] - ETA: 5s - loss: 7.6516 - accuracy: 0.5010
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6516 - accuracy: 0.5010
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6509 - accuracy: 0.5010
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6503 - accuracy: 0.5011
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6489 - accuracy: 0.5012
22592/25000 [==========================>...] - ETA: 4s - loss: 7.6517 - accuracy: 0.5010
22624/25000 [==========================>...] - ETA: 4s - loss: 7.6504 - accuracy: 0.5011
22656/25000 [==========================>...] - ETA: 4s - loss: 7.6483 - accuracy: 0.5012
22688/25000 [==========================>...] - ETA: 4s - loss: 7.6484 - accuracy: 0.5012
22720/25000 [==========================>...] - ETA: 4s - loss: 7.6470 - accuracy: 0.5013
22752/25000 [==========================>...] - ETA: 4s - loss: 7.6471 - accuracy: 0.5013
22784/25000 [==========================>...] - ETA: 4s - loss: 7.6491 - accuracy: 0.5011
22816/25000 [==========================>...] - ETA: 4s - loss: 7.6532 - accuracy: 0.5009
22848/25000 [==========================>...] - ETA: 4s - loss: 7.6552 - accuracy: 0.5007
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6559 - accuracy: 0.5007
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6546 - accuracy: 0.5008
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6559 - accuracy: 0.5007
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6593 - accuracy: 0.5005
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6586 - accuracy: 0.5005
23072/25000 [==========================>...] - ETA: 3s - loss: 7.6613 - accuracy: 0.5003
23104/25000 [==========================>...] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23136/25000 [==========================>...] - ETA: 3s - loss: 7.6633 - accuracy: 0.5002
23168/25000 [==========================>...] - ETA: 3s - loss: 7.6626 - accuracy: 0.5003
23200/25000 [==========================>...] - ETA: 3s - loss: 7.6633 - accuracy: 0.5002
23232/25000 [==========================>...] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23264/25000 [==========================>...] - ETA: 3s - loss: 7.6693 - accuracy: 0.4998
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
23552/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
23584/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
23616/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23648/25000 [===========================>..] - ETA: 2s - loss: 7.6627 - accuracy: 0.5003
23680/25000 [===========================>..] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
23712/25000 [===========================>..] - ETA: 2s - loss: 7.6614 - accuracy: 0.5003
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6628 - accuracy: 0.5003
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6608 - accuracy: 0.5004
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6596 - accuracy: 0.5005
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6596 - accuracy: 0.5005
24032/25000 [===========================>..] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
24064/25000 [===========================>..] - ETA: 1s - loss: 7.6596 - accuracy: 0.5005
24096/25000 [===========================>..] - ETA: 1s - loss: 7.6596 - accuracy: 0.5005
24128/25000 [===========================>..] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
24192/25000 [============================>.] - ETA: 1s - loss: 7.6609 - accuracy: 0.5004
24224/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24256/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24288/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24320/25000 [============================>.] - ETA: 1s - loss: 7.6609 - accuracy: 0.5004
24352/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
24384/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24416/25000 [============================>.] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
24448/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24480/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24512/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24544/25000 [============================>.] - ETA: 0s - loss: 7.6579 - accuracy: 0.5006
24576/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24608/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24768/25000 [============================>.] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24800/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24864/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24896/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6605 - accuracy: 0.5004
24960/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 60s 2ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

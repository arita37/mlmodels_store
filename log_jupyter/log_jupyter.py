
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
	Data preprocessing and feature engineering runtime = 0.25s ...
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
 40%|████      | 2/5 [00:54<01:21, 27.02s/it] 40%|████      | 2/5 [00:54<01:21, 27.02s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.0029222486823005105, 'embedding_size_factor': 1.062868882655795, 'layers.choice': 3, 'learning_rate': 0.004568828526103931, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00022706011724374898} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?g\xf0fP\xae)\x01X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\x01\x82\xcd/\x15NX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?r\xb6\xc3\x91\x9cA[X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?-\xc2\xdf\x8e:4\xc5u.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?g\xf0fP\xae)\x01X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\x01\x82\xcd/\x15NX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?r\xb6\xc3\x91\x9cA[X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?-\xc2\xdf\x8e:4\xc5u.' and reward: 0.3888
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 112.37500953674316
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 1, 'dropout_prob': 0.0029222486823005105, 'embedding_size_factor': 1.062868882655795, 'layers.choice': 3, 'learning_rate': 0.004568828526103931, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00022706011724374898}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the 5.75s of remaining time.
Ensemble size: 17
Ensemble weights: 
[0.82352941 0.17647059]
	0.3902	 = Validation accuracy score
	0.98s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 115.28s ...
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
 3366912/17464789 [====>.........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-23 15:20:32.850744: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 15:20:32.856122: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-05-23 15:20:32.856280: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b4acff3d80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 15:20:32.856295: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:54 - loss: 9.1041 - accuracy: 0.4062
   64/25000 [..............................] - ETA: 3:00 - loss: 6.9479 - accuracy: 0.5469
   96/25000 [..............................] - ETA: 2:22 - loss: 7.6666 - accuracy: 0.5000
  128/25000 [..............................] - ETA: 2:03 - loss: 8.0260 - accuracy: 0.4766
  160/25000 [..............................] - ETA: 1:52 - loss: 8.2416 - accuracy: 0.4625
  192/25000 [..............................] - ETA: 1:45 - loss: 8.3055 - accuracy: 0.4583
  224/25000 [..............................] - ETA: 1:42 - loss: 8.0089 - accuracy: 0.4777
  256/25000 [..............................] - ETA: 1:39 - loss: 8.0859 - accuracy: 0.4727
  288/25000 [..............................] - ETA: 1:36 - loss: 8.0925 - accuracy: 0.4722
  320/25000 [..............................] - ETA: 1:33 - loss: 8.1458 - accuracy: 0.4688
  352/25000 [..............................] - ETA: 1:31 - loss: 8.0587 - accuracy: 0.4744
  384/25000 [..............................] - ETA: 1:29 - loss: 7.9461 - accuracy: 0.4818
  416/25000 [..............................] - ETA: 1:28 - loss: 7.9246 - accuracy: 0.4832
  448/25000 [..............................] - ETA: 1:27 - loss: 8.0431 - accuracy: 0.4754
  480/25000 [..............................] - ETA: 1:26 - loss: 7.8263 - accuracy: 0.4896
  512/25000 [..............................] - ETA: 1:25 - loss: 7.6666 - accuracy: 0.5000
  544/25000 [..............................] - ETA: 1:23 - loss: 7.6948 - accuracy: 0.4982
  576/25000 [..............................] - ETA: 1:23 - loss: 7.8263 - accuracy: 0.4896
  608/25000 [..............................] - ETA: 1:22 - loss: 7.8684 - accuracy: 0.4868
  640/25000 [..............................] - ETA: 1:21 - loss: 7.8822 - accuracy: 0.4859
  672/25000 [..............................] - ETA: 1:21 - loss: 7.7351 - accuracy: 0.4955
  704/25000 [..............................] - ETA: 1:21 - loss: 7.7755 - accuracy: 0.4929
  736/25000 [..............................] - ETA: 1:20 - loss: 7.7083 - accuracy: 0.4973
  768/25000 [..............................] - ETA: 1:20 - loss: 7.6866 - accuracy: 0.4987
  800/25000 [..............................] - ETA: 1:20 - loss: 7.7241 - accuracy: 0.4963
  832/25000 [..............................] - ETA: 1:19 - loss: 7.7035 - accuracy: 0.4976
  864/25000 [>.............................] - ETA: 1:19 - loss: 7.6666 - accuracy: 0.5000
  896/25000 [>.............................] - ETA: 1:18 - loss: 7.5811 - accuracy: 0.5056
  928/25000 [>.............................] - ETA: 1:18 - loss: 7.5675 - accuracy: 0.5065
  960/25000 [>.............................] - ETA: 1:17 - loss: 7.6187 - accuracy: 0.5031
  992/25000 [>.............................] - ETA: 1:17 - loss: 7.6202 - accuracy: 0.5030
 1024/25000 [>.............................] - ETA: 1:16 - loss: 7.6816 - accuracy: 0.4990
 1056/25000 [>.............................] - ETA: 1:16 - loss: 7.7102 - accuracy: 0.4972
 1088/25000 [>.............................] - ETA: 1:16 - loss: 7.6384 - accuracy: 0.5018
 1120/25000 [>.............................] - ETA: 1:15 - loss: 7.6529 - accuracy: 0.5009
 1152/25000 [>.............................] - ETA: 1:15 - loss: 7.6533 - accuracy: 0.5009
 1184/25000 [>.............................] - ETA: 1:15 - loss: 7.6796 - accuracy: 0.4992
 1216/25000 [>.............................] - ETA: 1:14 - loss: 7.6540 - accuracy: 0.5008
 1248/25000 [>.............................] - ETA: 1:14 - loss: 7.5806 - accuracy: 0.5056
 1280/25000 [>.............................] - ETA: 1:14 - loss: 7.5947 - accuracy: 0.5047
 1312/25000 [>.............................] - ETA: 1:14 - loss: 7.5848 - accuracy: 0.5053
 1344/25000 [>.............................] - ETA: 1:13 - loss: 7.6096 - accuracy: 0.5037
 1376/25000 [>.............................] - ETA: 1:13 - loss: 7.6220 - accuracy: 0.5029
 1408/25000 [>.............................] - ETA: 1:13 - loss: 7.6339 - accuracy: 0.5021
 1440/25000 [>.............................] - ETA: 1:13 - loss: 7.6666 - accuracy: 0.5000
 1472/25000 [>.............................] - ETA: 1:12 - loss: 7.6354 - accuracy: 0.5020
 1504/25000 [>.............................] - ETA: 1:12 - loss: 7.6462 - accuracy: 0.5013
 1536/25000 [>.............................] - ETA: 1:12 - loss: 7.6467 - accuracy: 0.5013
 1568/25000 [>.............................] - ETA: 1:12 - loss: 7.6862 - accuracy: 0.4987
 1600/25000 [>.............................] - ETA: 1:12 - loss: 7.6954 - accuracy: 0.4981
 1632/25000 [>.............................] - ETA: 1:12 - loss: 7.7418 - accuracy: 0.4951
 1664/25000 [>.............................] - ETA: 1:11 - loss: 7.8141 - accuracy: 0.4904
 1696/25000 [=>............................] - ETA: 1:11 - loss: 7.8203 - accuracy: 0.4900
 1728/25000 [=>............................] - ETA: 1:11 - loss: 7.7997 - accuracy: 0.4913
 1760/25000 [=>............................] - ETA: 1:11 - loss: 7.7886 - accuracy: 0.4920
 1792/25000 [=>............................] - ETA: 1:11 - loss: 7.7693 - accuracy: 0.4933
 1824/25000 [=>............................] - ETA: 1:11 - loss: 7.7675 - accuracy: 0.4934
 1856/25000 [=>............................] - ETA: 1:11 - loss: 7.7492 - accuracy: 0.4946
 1888/25000 [=>............................] - ETA: 1:11 - loss: 7.7478 - accuracy: 0.4947
 1920/25000 [=>............................] - ETA: 1:10 - loss: 7.7385 - accuracy: 0.4953
 1952/25000 [=>............................] - ETA: 1:10 - loss: 7.7138 - accuracy: 0.4969
 1984/25000 [=>............................] - ETA: 1:10 - loss: 7.7362 - accuracy: 0.4955
 2016/25000 [=>............................] - ETA: 1:10 - loss: 7.7883 - accuracy: 0.4921
 2048/25000 [=>............................] - ETA: 1:10 - loss: 7.7565 - accuracy: 0.4941
 2080/25000 [=>............................] - ETA: 1:10 - loss: 7.7551 - accuracy: 0.4942
 2112/25000 [=>............................] - ETA: 1:10 - loss: 7.7465 - accuracy: 0.4948
 2144/25000 [=>............................] - ETA: 1:09 - loss: 7.7739 - accuracy: 0.4930
 2176/25000 [=>............................] - ETA: 1:09 - loss: 7.7653 - accuracy: 0.4936
 2208/25000 [=>............................] - ETA: 1:09 - loss: 7.7777 - accuracy: 0.4928
 2240/25000 [=>............................] - ETA: 1:09 - loss: 7.7556 - accuracy: 0.4942
 2272/25000 [=>............................] - ETA: 1:09 - loss: 7.7544 - accuracy: 0.4943
 2304/25000 [=>............................] - ETA: 1:09 - loss: 7.7664 - accuracy: 0.4935
 2336/25000 [=>............................] - ETA: 1:09 - loss: 7.7454 - accuracy: 0.4949
 2368/25000 [=>............................] - ETA: 1:08 - loss: 7.7184 - accuracy: 0.4966
 2400/25000 [=>............................] - ETA: 1:08 - loss: 7.7305 - accuracy: 0.4958
 2432/25000 [=>............................] - ETA: 1:08 - loss: 7.7171 - accuracy: 0.4967
 2464/25000 [=>............................] - ETA: 1:08 - loss: 7.7288 - accuracy: 0.4959
 2496/25000 [=>............................] - ETA: 1:08 - loss: 7.7158 - accuracy: 0.4968
 2528/25000 [==>...........................] - ETA: 1:08 - loss: 7.7091 - accuracy: 0.4972
 2560/25000 [==>...........................] - ETA: 1:08 - loss: 7.6966 - accuracy: 0.4980
 2592/25000 [==>...........................] - ETA: 1:08 - loss: 7.6903 - accuracy: 0.4985
 2624/25000 [==>...........................] - ETA: 1:07 - loss: 7.6608 - accuracy: 0.5004
 2656/25000 [==>...........................] - ETA: 1:07 - loss: 7.6782 - accuracy: 0.4992
 2688/25000 [==>...........................] - ETA: 1:07 - loss: 7.6723 - accuracy: 0.4996
 2720/25000 [==>...........................] - ETA: 1:07 - loss: 7.6723 - accuracy: 0.4996
 2752/25000 [==>...........................] - ETA: 1:07 - loss: 7.6666 - accuracy: 0.5000
 2784/25000 [==>...........................] - ETA: 1:07 - loss: 7.6666 - accuracy: 0.5000
 2816/25000 [==>...........................] - ETA: 1:07 - loss: 7.6775 - accuracy: 0.4993
 2848/25000 [==>...........................] - ETA: 1:06 - loss: 7.6774 - accuracy: 0.4993
 2880/25000 [==>...........................] - ETA: 1:06 - loss: 7.6879 - accuracy: 0.4986
 2912/25000 [==>...........................] - ETA: 1:06 - loss: 7.6719 - accuracy: 0.4997
 2944/25000 [==>...........................] - ETA: 1:06 - loss: 7.6562 - accuracy: 0.5007
 2976/25000 [==>...........................] - ETA: 1:06 - loss: 7.6666 - accuracy: 0.5000
 3008/25000 [==>...........................] - ETA: 1:06 - loss: 7.6513 - accuracy: 0.5010
 3040/25000 [==>...........................] - ETA: 1:05 - loss: 7.6565 - accuracy: 0.5007
 3072/25000 [==>...........................] - ETA: 1:05 - loss: 7.6616 - accuracy: 0.5003
 3104/25000 [==>...........................] - ETA: 1:05 - loss: 7.6666 - accuracy: 0.5000
 3136/25000 [==>...........................] - ETA: 1:05 - loss: 7.6960 - accuracy: 0.4981
 3168/25000 [==>...........................] - ETA: 1:05 - loss: 7.7053 - accuracy: 0.4975
 3200/25000 [==>...........................] - ETA: 1:05 - loss: 7.6810 - accuracy: 0.4991
 3232/25000 [==>...........................] - ETA: 1:05 - loss: 7.6998 - accuracy: 0.4978
 3264/25000 [==>...........................] - ETA: 1:05 - loss: 7.6807 - accuracy: 0.4991
 3296/25000 [==>...........................] - ETA: 1:04 - loss: 7.6852 - accuracy: 0.4988
 3328/25000 [==>...........................] - ETA: 1:04 - loss: 7.6943 - accuracy: 0.4982
 3360/25000 [===>..........................] - ETA: 1:04 - loss: 7.6894 - accuracy: 0.4985
 3392/25000 [===>..........................] - ETA: 1:04 - loss: 7.6847 - accuracy: 0.4988
 3424/25000 [===>..........................] - ETA: 1:04 - loss: 7.6487 - accuracy: 0.5012
 3456/25000 [===>..........................] - ETA: 1:04 - loss: 7.6267 - accuracy: 0.5026
 3488/25000 [===>..........................] - ETA: 1:04 - loss: 7.6227 - accuracy: 0.5029
 3520/25000 [===>..........................] - ETA: 1:03 - loss: 7.6274 - accuracy: 0.5026
 3552/25000 [===>..........................] - ETA: 1:03 - loss: 7.6407 - accuracy: 0.5017
 3584/25000 [===>..........................] - ETA: 1:03 - loss: 7.6452 - accuracy: 0.5014
 3616/25000 [===>..........................] - ETA: 1:03 - loss: 7.6242 - accuracy: 0.5028
 3648/25000 [===>..........................] - ETA: 1:03 - loss: 7.6288 - accuracy: 0.5025
 3680/25000 [===>..........................] - ETA: 1:03 - loss: 7.6416 - accuracy: 0.5016
 3712/25000 [===>..........................] - ETA: 1:03 - loss: 7.6542 - accuracy: 0.5008
 3744/25000 [===>..........................] - ETA: 1:03 - loss: 7.6420 - accuracy: 0.5016
 3776/25000 [===>..........................] - ETA: 1:02 - loss: 7.6544 - accuracy: 0.5008
 3808/25000 [===>..........................] - ETA: 1:02 - loss: 7.6505 - accuracy: 0.5011
 3840/25000 [===>..........................] - ETA: 1:02 - loss: 7.6626 - accuracy: 0.5003
 3872/25000 [===>..........................] - ETA: 1:02 - loss: 7.6547 - accuracy: 0.5008
 3904/25000 [===>..........................] - ETA: 1:02 - loss: 7.6588 - accuracy: 0.5005
 3936/25000 [===>..........................] - ETA: 1:02 - loss: 7.6549 - accuracy: 0.5008
 3968/25000 [===>..........................] - ETA: 1:02 - loss: 7.6666 - accuracy: 0.5000
 4000/25000 [===>..........................] - ETA: 1:02 - loss: 7.6513 - accuracy: 0.5010
 4032/25000 [===>..........................] - ETA: 1:01 - loss: 7.6476 - accuracy: 0.5012
 4064/25000 [===>..........................] - ETA: 1:01 - loss: 7.6440 - accuracy: 0.5015
 4096/25000 [===>..........................] - ETA: 1:01 - loss: 7.6554 - accuracy: 0.5007
 4128/25000 [===>..........................] - ETA: 1:01 - loss: 7.6555 - accuracy: 0.5007
 4160/25000 [===>..........................] - ETA: 1:01 - loss: 7.6556 - accuracy: 0.5007
 4192/25000 [====>.........................] - ETA: 1:01 - loss: 7.6630 - accuracy: 0.5002
 4224/25000 [====>.........................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 4256/25000 [====>.........................] - ETA: 1:01 - loss: 7.6702 - accuracy: 0.4998
 4288/25000 [====>.........................] - ETA: 1:01 - loss: 7.6809 - accuracy: 0.4991
 4320/25000 [====>.........................] - ETA: 1:00 - loss: 7.6808 - accuracy: 0.4991
 4352/25000 [====>.........................] - ETA: 1:00 - loss: 7.6701 - accuracy: 0.4998
 4384/25000 [====>.........................] - ETA: 1:00 - loss: 7.6841 - accuracy: 0.4989
 4416/25000 [====>.........................] - ETA: 1:00 - loss: 7.6840 - accuracy: 0.4989
 4448/25000 [====>.........................] - ETA: 1:00 - loss: 7.6735 - accuracy: 0.4996
 4480/25000 [====>.........................] - ETA: 1:00 - loss: 7.6700 - accuracy: 0.4998
 4512/25000 [====>.........................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 4544/25000 [====>.........................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 4576/25000 [====>.........................] - ETA: 59s - loss: 7.6633 - accuracy: 0.5002 
 4608/25000 [====>.........................] - ETA: 59s - loss: 7.6566 - accuracy: 0.5007
 4640/25000 [====>.........................] - ETA: 59s - loss: 7.6732 - accuracy: 0.4996
 4672/25000 [====>.........................] - ETA: 59s - loss: 7.6863 - accuracy: 0.4987
 4704/25000 [====>.........................] - ETA: 59s - loss: 7.6862 - accuracy: 0.4987
 4736/25000 [====>.........................] - ETA: 59s - loss: 7.6925 - accuracy: 0.4983
 4768/25000 [====>.........................] - ETA: 59s - loss: 7.6956 - accuracy: 0.4981
 4800/25000 [====>.........................] - ETA: 59s - loss: 7.6922 - accuracy: 0.4983
 4832/25000 [====>.........................] - ETA: 59s - loss: 7.6920 - accuracy: 0.4983
 4864/25000 [====>.........................] - ETA: 58s - loss: 7.6887 - accuracy: 0.4986
 4896/25000 [====>.........................] - ETA: 58s - loss: 7.6854 - accuracy: 0.4988
 4928/25000 [====>.........................] - ETA: 58s - loss: 7.6946 - accuracy: 0.4982
 4960/25000 [====>.........................] - ETA: 58s - loss: 7.6728 - accuracy: 0.4996
 4992/25000 [====>.........................] - ETA: 58s - loss: 7.6758 - accuracy: 0.4994
 5024/25000 [=====>........................] - ETA: 58s - loss: 7.6849 - accuracy: 0.4988
 5056/25000 [=====>........................] - ETA: 58s - loss: 7.7030 - accuracy: 0.4976
 5088/25000 [=====>........................] - ETA: 58s - loss: 7.6998 - accuracy: 0.4978
 5120/25000 [=====>........................] - ETA: 57s - loss: 7.7056 - accuracy: 0.4975
 5152/25000 [=====>........................] - ETA: 57s - loss: 7.7053 - accuracy: 0.4975
 5184/25000 [=====>........................] - ETA: 57s - loss: 7.6962 - accuracy: 0.4981
 5216/25000 [=====>........................] - ETA: 57s - loss: 7.6931 - accuracy: 0.4983
 5248/25000 [=====>........................] - ETA: 57s - loss: 7.6929 - accuracy: 0.4983
 5280/25000 [=====>........................] - ETA: 57s - loss: 7.6957 - accuracy: 0.4981
 5312/25000 [=====>........................] - ETA: 57s - loss: 7.6955 - accuracy: 0.4981
 5344/25000 [=====>........................] - ETA: 57s - loss: 7.7039 - accuracy: 0.4976
 5376/25000 [=====>........................] - ETA: 57s - loss: 7.7065 - accuracy: 0.4974
 5408/25000 [=====>........................] - ETA: 56s - loss: 7.6836 - accuracy: 0.4989
 5440/25000 [=====>........................] - ETA: 56s - loss: 7.6779 - accuracy: 0.4993
 5472/25000 [=====>........................] - ETA: 56s - loss: 7.6806 - accuracy: 0.4991
 5504/25000 [=====>........................] - ETA: 56s - loss: 7.6861 - accuracy: 0.4987
 5536/25000 [=====>........................] - ETA: 56s - loss: 7.6722 - accuracy: 0.4996
 5568/25000 [=====>........................] - ETA: 56s - loss: 7.6694 - accuracy: 0.4998
 5600/25000 [=====>........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 5632/25000 [=====>........................] - ETA: 56s - loss: 7.6612 - accuracy: 0.5004
 5664/25000 [=====>........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 5696/25000 [=====>........................] - ETA: 56s - loss: 7.6639 - accuracy: 0.5002
 5728/25000 [=====>........................] - ETA: 55s - loss: 7.6693 - accuracy: 0.4998
 5760/25000 [=====>........................] - ETA: 55s - loss: 7.6560 - accuracy: 0.5007
 5792/25000 [=====>........................] - ETA: 55s - loss: 7.6534 - accuracy: 0.5009
 5824/25000 [=====>........................] - ETA: 55s - loss: 7.6640 - accuracy: 0.5002
 5856/25000 [======>.......................] - ETA: 55s - loss: 7.6692 - accuracy: 0.4998
 5888/25000 [======>.......................] - ETA: 55s - loss: 7.6770 - accuracy: 0.4993
 5920/25000 [======>.......................] - ETA: 55s - loss: 7.6796 - accuracy: 0.4992
 5952/25000 [======>.......................] - ETA: 55s - loss: 7.6847 - accuracy: 0.4988
 5984/25000 [======>.......................] - ETA: 54s - loss: 7.6897 - accuracy: 0.4985
 6016/25000 [======>.......................] - ETA: 54s - loss: 7.6998 - accuracy: 0.4978
 6048/25000 [======>.......................] - ETA: 54s - loss: 7.7021 - accuracy: 0.4977
 6080/25000 [======>.......................] - ETA: 54s - loss: 7.6994 - accuracy: 0.4979
 6112/25000 [======>.......................] - ETA: 54s - loss: 7.7068 - accuracy: 0.4974
 6144/25000 [======>.......................] - ETA: 54s - loss: 7.7115 - accuracy: 0.4971
 6176/25000 [======>.......................] - ETA: 54s - loss: 7.6989 - accuracy: 0.4979
 6208/25000 [======>.......................] - ETA: 54s - loss: 7.6987 - accuracy: 0.4979
 6240/25000 [======>.......................] - ETA: 54s - loss: 7.7133 - accuracy: 0.4970
 6272/25000 [======>.......................] - ETA: 54s - loss: 7.7155 - accuracy: 0.4968
 6304/25000 [======>.......................] - ETA: 53s - loss: 7.7104 - accuracy: 0.4971
 6336/25000 [======>.......................] - ETA: 53s - loss: 7.7005 - accuracy: 0.4978
 6368/25000 [======>.......................] - ETA: 53s - loss: 7.7027 - accuracy: 0.4976
 6400/25000 [======>.......................] - ETA: 53s - loss: 7.7097 - accuracy: 0.4972
 6432/25000 [======>.......................] - ETA: 53s - loss: 7.7048 - accuracy: 0.4975
 6464/25000 [======>.......................] - ETA: 53s - loss: 7.7093 - accuracy: 0.4972
 6496/25000 [======>.......................] - ETA: 53s - loss: 7.7185 - accuracy: 0.4966
 6528/25000 [======>.......................] - ETA: 53s - loss: 7.7347 - accuracy: 0.4956
 6560/25000 [======>.......................] - ETA: 53s - loss: 7.7461 - accuracy: 0.4948
 6592/25000 [======>.......................] - ETA: 52s - loss: 7.7504 - accuracy: 0.4945
 6624/25000 [======>.......................] - ETA: 52s - loss: 7.7453 - accuracy: 0.4949
 6656/25000 [======>.......................] - ETA: 52s - loss: 7.7519 - accuracy: 0.4944
 6688/25000 [=======>......................] - ETA: 52s - loss: 7.7606 - accuracy: 0.4939
 6720/25000 [=======>......................] - ETA: 52s - loss: 7.7693 - accuracy: 0.4933
 6752/25000 [=======>......................] - ETA: 52s - loss: 7.7643 - accuracy: 0.4936
 6784/25000 [=======>......................] - ETA: 52s - loss: 7.7638 - accuracy: 0.4937
 6816/25000 [=======>......................] - ETA: 52s - loss: 7.7544 - accuracy: 0.4943
 6848/25000 [=======>......................] - ETA: 52s - loss: 7.7495 - accuracy: 0.4946
 6880/25000 [=======>......................] - ETA: 51s - loss: 7.7446 - accuracy: 0.4949
 6912/25000 [=======>......................] - ETA: 51s - loss: 7.7465 - accuracy: 0.4948
 6944/25000 [=======>......................] - ETA: 51s - loss: 7.7395 - accuracy: 0.4952
 6976/25000 [=======>......................] - ETA: 51s - loss: 7.7457 - accuracy: 0.4948
 7008/25000 [=======>......................] - ETA: 51s - loss: 7.7476 - accuracy: 0.4947
 7040/25000 [=======>......................] - ETA: 51s - loss: 7.7494 - accuracy: 0.4946
 7072/25000 [=======>......................] - ETA: 51s - loss: 7.7447 - accuracy: 0.4949
 7104/25000 [=======>......................] - ETA: 51s - loss: 7.7422 - accuracy: 0.4951
 7136/25000 [=======>......................] - ETA: 51s - loss: 7.7483 - accuracy: 0.4947
 7168/25000 [=======>......................] - ETA: 51s - loss: 7.7479 - accuracy: 0.4947
 7200/25000 [=======>......................] - ETA: 50s - loss: 7.7454 - accuracy: 0.4949
 7232/25000 [=======>......................] - ETA: 50s - loss: 7.7578 - accuracy: 0.4941
 7264/25000 [=======>......................] - ETA: 50s - loss: 7.7574 - accuracy: 0.4941
 7296/25000 [=======>......................] - ETA: 50s - loss: 7.7507 - accuracy: 0.4945
 7328/25000 [=======>......................] - ETA: 50s - loss: 7.7566 - accuracy: 0.4941
 7360/25000 [=======>......................] - ETA: 50s - loss: 7.7645 - accuracy: 0.4936
 7392/25000 [=======>......................] - ETA: 50s - loss: 7.7579 - accuracy: 0.4940
 7424/25000 [=======>......................] - ETA: 50s - loss: 7.7616 - accuracy: 0.4938
 7456/25000 [=======>......................] - ETA: 50s - loss: 7.7715 - accuracy: 0.4932
 7488/25000 [=======>......................] - ETA: 50s - loss: 7.7731 - accuracy: 0.4931
 7520/25000 [========>.....................] - ETA: 49s - loss: 7.7584 - accuracy: 0.4940
 7552/25000 [========>.....................] - ETA: 49s - loss: 7.7641 - accuracy: 0.4936
 7584/25000 [========>.....................] - ETA: 49s - loss: 7.7677 - accuracy: 0.4934
 7616/25000 [========>.....................] - ETA: 49s - loss: 7.7633 - accuracy: 0.4937
 7648/25000 [========>.....................] - ETA: 49s - loss: 7.7629 - accuracy: 0.4937
 7680/25000 [========>.....................] - ETA: 49s - loss: 7.7565 - accuracy: 0.4941
 7712/25000 [========>.....................] - ETA: 49s - loss: 7.7521 - accuracy: 0.4944
 7744/25000 [========>.....................] - ETA: 49s - loss: 7.7537 - accuracy: 0.4943
 7776/25000 [========>.....................] - ETA: 49s - loss: 7.7534 - accuracy: 0.4943
 7808/25000 [========>.....................] - ETA: 49s - loss: 7.7570 - accuracy: 0.4941
 7840/25000 [========>.....................] - ETA: 48s - loss: 7.7566 - accuracy: 0.4941
 7872/25000 [========>.....................] - ETA: 48s - loss: 7.7660 - accuracy: 0.4935
 7904/25000 [========>.....................] - ETA: 48s - loss: 7.7636 - accuracy: 0.4937
 7936/25000 [========>.....................] - ETA: 48s - loss: 7.7632 - accuracy: 0.4937
 7968/25000 [========>.....................] - ETA: 48s - loss: 7.7628 - accuracy: 0.4937
 8000/25000 [========>.....................] - ETA: 48s - loss: 7.7625 - accuracy: 0.4938
 8032/25000 [========>.....................] - ETA: 48s - loss: 7.7621 - accuracy: 0.4938
 8064/25000 [========>.....................] - ETA: 48s - loss: 7.7560 - accuracy: 0.4942
 8096/25000 [========>.....................] - ETA: 48s - loss: 7.7537 - accuracy: 0.4943
 8128/25000 [========>.....................] - ETA: 48s - loss: 7.7515 - accuracy: 0.4945
 8160/25000 [========>.....................] - ETA: 47s - loss: 7.7512 - accuracy: 0.4945
 8192/25000 [========>.....................] - ETA: 47s - loss: 7.7452 - accuracy: 0.4949
 8224/25000 [========>.....................] - ETA: 47s - loss: 7.7505 - accuracy: 0.4945
 8256/25000 [========>.....................] - ETA: 47s - loss: 7.7558 - accuracy: 0.4942
 8288/25000 [========>.....................] - ETA: 47s - loss: 7.7591 - accuracy: 0.4940
 8320/25000 [========>.....................] - ETA: 47s - loss: 7.7532 - accuracy: 0.4944
 8352/25000 [=========>....................] - ETA: 47s - loss: 7.7492 - accuracy: 0.4946
 8384/25000 [=========>....................] - ETA: 47s - loss: 7.7562 - accuracy: 0.4942
 8416/25000 [=========>....................] - ETA: 47s - loss: 7.7614 - accuracy: 0.4938
 8448/25000 [=========>....................] - ETA: 47s - loss: 7.7683 - accuracy: 0.4934
 8480/25000 [=========>....................] - ETA: 47s - loss: 7.7751 - accuracy: 0.4929
 8512/25000 [=========>....................] - ETA: 46s - loss: 7.7837 - accuracy: 0.4924
 8544/25000 [=========>....................] - ETA: 46s - loss: 7.7815 - accuracy: 0.4925
 8576/25000 [=========>....................] - ETA: 46s - loss: 7.7793 - accuracy: 0.4927
 8608/25000 [=========>....................] - ETA: 46s - loss: 7.7824 - accuracy: 0.4924
 8640/25000 [=========>....................] - ETA: 46s - loss: 7.7855 - accuracy: 0.4922
 8672/25000 [=========>....................] - ETA: 46s - loss: 7.7780 - accuracy: 0.4927
 8704/25000 [=========>....................] - ETA: 46s - loss: 7.7776 - accuracy: 0.4928
 8736/25000 [=========>....................] - ETA: 46s - loss: 7.7790 - accuracy: 0.4927
 8768/25000 [=========>....................] - ETA: 46s - loss: 7.7820 - accuracy: 0.4925
 8800/25000 [=========>....................] - ETA: 45s - loss: 7.7834 - accuracy: 0.4924
 8832/25000 [=========>....................] - ETA: 45s - loss: 7.7829 - accuracy: 0.4924
 8864/25000 [=========>....................] - ETA: 45s - loss: 7.7894 - accuracy: 0.4920
 8896/25000 [=========>....................] - ETA: 45s - loss: 7.7838 - accuracy: 0.4924
 8928/25000 [=========>....................] - ETA: 45s - loss: 7.7868 - accuracy: 0.4922
 8960/25000 [=========>....................] - ETA: 45s - loss: 7.7779 - accuracy: 0.4927
 8992/25000 [=========>....................] - ETA: 45s - loss: 7.7792 - accuracy: 0.4927
 9024/25000 [=========>....................] - ETA: 45s - loss: 7.7771 - accuracy: 0.4928
 9056/25000 [=========>....................] - ETA: 45s - loss: 7.7801 - accuracy: 0.4926
 9088/25000 [=========>....................] - ETA: 45s - loss: 7.7780 - accuracy: 0.4927
 9120/25000 [=========>....................] - ETA: 45s - loss: 7.7809 - accuracy: 0.4925
 9152/25000 [=========>....................] - ETA: 44s - loss: 7.7889 - accuracy: 0.4920
 9184/25000 [==========>...................] - ETA: 44s - loss: 7.7968 - accuracy: 0.4915
 9216/25000 [==========>...................] - ETA: 44s - loss: 7.7997 - accuracy: 0.4913
 9248/25000 [==========>...................] - ETA: 44s - loss: 7.7926 - accuracy: 0.4918
 9280/25000 [==========>...................] - ETA: 44s - loss: 7.7972 - accuracy: 0.4915
 9312/25000 [==========>...................] - ETA: 44s - loss: 7.7918 - accuracy: 0.4918
 9344/25000 [==========>...................] - ETA: 44s - loss: 7.7946 - accuracy: 0.4917
 9376/25000 [==========>...................] - ETA: 44s - loss: 7.7893 - accuracy: 0.4920
 9408/25000 [==========>...................] - ETA: 44s - loss: 7.7970 - accuracy: 0.4915
 9440/25000 [==========>...................] - ETA: 44s - loss: 7.8047 - accuracy: 0.4910
 9472/25000 [==========>...................] - ETA: 43s - loss: 7.8107 - accuracy: 0.4906
 9504/25000 [==========>...................] - ETA: 43s - loss: 7.8150 - accuracy: 0.4903
 9536/25000 [==========>...................] - ETA: 43s - loss: 7.8162 - accuracy: 0.4902
 9568/25000 [==========>...................] - ETA: 43s - loss: 7.8205 - accuracy: 0.4900
 9600/25000 [==========>...................] - ETA: 43s - loss: 7.8168 - accuracy: 0.4902
 9632/25000 [==========>...................] - ETA: 43s - loss: 7.8179 - accuracy: 0.4901
 9664/25000 [==========>...................] - ETA: 43s - loss: 7.8205 - accuracy: 0.4900
 9696/25000 [==========>...................] - ETA: 43s - loss: 7.8248 - accuracy: 0.4897
 9728/25000 [==========>...................] - ETA: 43s - loss: 7.8242 - accuracy: 0.4897
 9760/25000 [==========>...................] - ETA: 43s - loss: 7.8222 - accuracy: 0.4899
 9792/25000 [==========>...................] - ETA: 42s - loss: 7.8279 - accuracy: 0.4895
 9824/25000 [==========>...................] - ETA: 42s - loss: 7.8227 - accuracy: 0.4898
 9856/25000 [==========>...................] - ETA: 42s - loss: 7.8160 - accuracy: 0.4903
 9888/25000 [==========>...................] - ETA: 42s - loss: 7.8155 - accuracy: 0.4903
 9920/25000 [==========>...................] - ETA: 42s - loss: 7.8181 - accuracy: 0.4901
 9952/25000 [==========>...................] - ETA: 42s - loss: 7.8130 - accuracy: 0.4905
 9984/25000 [==========>...................] - ETA: 42s - loss: 7.8141 - accuracy: 0.4904
10016/25000 [===========>..................] - ETA: 42s - loss: 7.8136 - accuracy: 0.4904
10048/25000 [===========>..................] - ETA: 42s - loss: 7.8085 - accuracy: 0.4907
10080/25000 [===========>..................] - ETA: 42s - loss: 7.8111 - accuracy: 0.4906
10112/25000 [===========>..................] - ETA: 42s - loss: 7.8092 - accuracy: 0.4907
10144/25000 [===========>..................] - ETA: 41s - loss: 7.8042 - accuracy: 0.4910
10176/25000 [===========>..................] - ETA: 41s - loss: 7.8052 - accuracy: 0.4910
10208/25000 [===========>..................] - ETA: 41s - loss: 7.8018 - accuracy: 0.4912
10240/25000 [===========>..................] - ETA: 41s - loss: 7.8044 - accuracy: 0.4910
10272/25000 [===========>..................] - ETA: 41s - loss: 7.7965 - accuracy: 0.4915
10304/25000 [===========>..................] - ETA: 41s - loss: 7.7916 - accuracy: 0.4918
10336/25000 [===========>..................] - ETA: 41s - loss: 7.7868 - accuracy: 0.4922
10368/25000 [===========>..................] - ETA: 41s - loss: 7.7820 - accuracy: 0.4925
10400/25000 [===========>..................] - ETA: 41s - loss: 7.7816 - accuracy: 0.4925
10432/25000 [===========>..................] - ETA: 41s - loss: 7.7769 - accuracy: 0.4928
10464/25000 [===========>..................] - ETA: 40s - loss: 7.7707 - accuracy: 0.4932
10496/25000 [===========>..................] - ETA: 40s - loss: 7.7718 - accuracy: 0.4931
10528/25000 [===========>..................] - ETA: 40s - loss: 7.7802 - accuracy: 0.4926
10560/25000 [===========>..................] - ETA: 40s - loss: 7.7813 - accuracy: 0.4925
10592/25000 [===========>..................] - ETA: 40s - loss: 7.7824 - accuracy: 0.4924
10624/25000 [===========>..................] - ETA: 40s - loss: 7.7850 - accuracy: 0.4923
10656/25000 [===========>..................] - ETA: 40s - loss: 7.7918 - accuracy: 0.4918
10688/25000 [===========>..................] - ETA: 40s - loss: 7.7943 - accuracy: 0.4917
10720/25000 [===========>..................] - ETA: 40s - loss: 7.8025 - accuracy: 0.4911
10752/25000 [===========>..................] - ETA: 40s - loss: 7.8035 - accuracy: 0.4911
10784/25000 [===========>..................] - ETA: 40s - loss: 7.8017 - accuracy: 0.4912
10816/25000 [===========>..................] - ETA: 39s - loss: 7.7956 - accuracy: 0.4916
10848/25000 [============>.................] - ETA: 39s - loss: 7.7924 - accuracy: 0.4918
10880/25000 [============>.................] - ETA: 39s - loss: 7.7906 - accuracy: 0.4919
10912/25000 [============>.................] - ETA: 39s - loss: 7.7945 - accuracy: 0.4917
10944/25000 [============>.................] - ETA: 39s - loss: 7.7955 - accuracy: 0.4916
10976/25000 [============>.................] - ETA: 39s - loss: 7.7937 - accuracy: 0.4917
11008/25000 [============>.................] - ETA: 39s - loss: 7.7976 - accuracy: 0.4915
11040/25000 [============>.................] - ETA: 39s - loss: 7.7958 - accuracy: 0.4916
11072/25000 [============>.................] - ETA: 39s - loss: 7.7954 - accuracy: 0.4916
11104/25000 [============>.................] - ETA: 39s - loss: 7.7950 - accuracy: 0.4916
11136/25000 [============>.................] - ETA: 39s - loss: 7.7919 - accuracy: 0.4918
11168/25000 [============>.................] - ETA: 38s - loss: 7.7916 - accuracy: 0.4919
11200/25000 [============>.................] - ETA: 38s - loss: 7.7967 - accuracy: 0.4915
11232/25000 [============>.................] - ETA: 38s - loss: 7.8059 - accuracy: 0.4909
11264/25000 [============>.................] - ETA: 38s - loss: 7.8109 - accuracy: 0.4906
11296/25000 [============>.................] - ETA: 38s - loss: 7.8119 - accuracy: 0.4905
11328/25000 [============>.................] - ETA: 38s - loss: 7.8142 - accuracy: 0.4904
11360/25000 [============>.................] - ETA: 38s - loss: 7.8083 - accuracy: 0.4908
11392/25000 [============>.................] - ETA: 38s - loss: 7.8106 - accuracy: 0.4906
11424/25000 [============>.................] - ETA: 38s - loss: 7.8075 - accuracy: 0.4908
11456/25000 [============>.................] - ETA: 38s - loss: 7.8005 - accuracy: 0.4913
11488/25000 [============>.................] - ETA: 37s - loss: 7.8028 - accuracy: 0.4911
11520/25000 [============>.................] - ETA: 37s - loss: 7.8090 - accuracy: 0.4907
11552/25000 [============>.................] - ETA: 37s - loss: 7.8086 - accuracy: 0.4907
11584/25000 [============>.................] - ETA: 37s - loss: 7.8122 - accuracy: 0.4905
11616/25000 [============>.................] - ETA: 37s - loss: 7.8118 - accuracy: 0.4905
11648/25000 [============>.................] - ETA: 37s - loss: 7.8062 - accuracy: 0.4909
11680/25000 [=============>................] - ETA: 37s - loss: 7.8097 - accuracy: 0.4907
11712/25000 [=============>................] - ETA: 37s - loss: 7.8067 - accuracy: 0.4909
11744/25000 [=============>................] - ETA: 37s - loss: 7.8115 - accuracy: 0.4905
11776/25000 [=============>................] - ETA: 37s - loss: 7.8138 - accuracy: 0.4904
11808/25000 [=============>................] - ETA: 37s - loss: 7.8121 - accuracy: 0.4905
11840/25000 [=============>................] - ETA: 36s - loss: 7.8130 - accuracy: 0.4905
11872/25000 [=============>................] - ETA: 36s - loss: 7.8139 - accuracy: 0.4904
11904/25000 [=============>................] - ETA: 36s - loss: 7.8173 - accuracy: 0.4902
11936/25000 [=============>................] - ETA: 36s - loss: 7.8221 - accuracy: 0.4899
11968/25000 [=============>................] - ETA: 36s - loss: 7.8255 - accuracy: 0.4896
12000/25000 [=============>................] - ETA: 36s - loss: 7.8225 - accuracy: 0.4898
12032/25000 [=============>................] - ETA: 36s - loss: 7.8195 - accuracy: 0.4900
12064/25000 [=============>................] - ETA: 36s - loss: 7.8166 - accuracy: 0.4902
12096/25000 [=============>................] - ETA: 36s - loss: 7.8124 - accuracy: 0.4905
12128/25000 [=============>................] - ETA: 36s - loss: 7.8107 - accuracy: 0.4906
12160/25000 [=============>................] - ETA: 36s - loss: 7.8091 - accuracy: 0.4907
12192/25000 [=============>................] - ETA: 35s - loss: 7.8100 - accuracy: 0.4906
12224/25000 [=============>................] - ETA: 35s - loss: 7.8096 - accuracy: 0.4907
12256/25000 [=============>................] - ETA: 35s - loss: 7.8105 - accuracy: 0.4906
12288/25000 [=============>................] - ETA: 35s - loss: 7.8114 - accuracy: 0.4906
12320/25000 [=============>................] - ETA: 35s - loss: 7.8073 - accuracy: 0.4908
12352/25000 [=============>................] - ETA: 35s - loss: 7.8106 - accuracy: 0.4906
12384/25000 [=============>................] - ETA: 35s - loss: 7.8140 - accuracy: 0.4904
12416/25000 [=============>................] - ETA: 35s - loss: 7.8086 - accuracy: 0.4907
12448/25000 [=============>................] - ETA: 35s - loss: 7.8083 - accuracy: 0.4908
12480/25000 [=============>................] - ETA: 35s - loss: 7.8067 - accuracy: 0.4909
12512/25000 [==============>...............] - ETA: 34s - loss: 7.8088 - accuracy: 0.4907
12544/25000 [==============>...............] - ETA: 34s - loss: 7.8047 - accuracy: 0.4910
12576/25000 [==============>...............] - ETA: 34s - loss: 7.8068 - accuracy: 0.4909
12608/25000 [==============>...............] - ETA: 34s - loss: 7.7992 - accuracy: 0.4914
12640/25000 [==============>...............] - ETA: 34s - loss: 7.7988 - accuracy: 0.4914
12672/25000 [==============>...............] - ETA: 34s - loss: 7.7997 - accuracy: 0.4913
12704/25000 [==============>...............] - ETA: 34s - loss: 7.8006 - accuracy: 0.4913
12736/25000 [==============>...............] - ETA: 34s - loss: 7.7978 - accuracy: 0.4914
12768/25000 [==============>...............] - ETA: 34s - loss: 7.7975 - accuracy: 0.4915
12800/25000 [==============>...............] - ETA: 34s - loss: 7.7984 - accuracy: 0.4914
12832/25000 [==============>...............] - ETA: 34s - loss: 7.7957 - accuracy: 0.4916
12864/25000 [==============>...............] - ETA: 33s - loss: 7.7977 - accuracy: 0.4914
12896/25000 [==============>...............] - ETA: 33s - loss: 7.8010 - accuracy: 0.4912
12928/25000 [==============>...............] - ETA: 33s - loss: 7.7935 - accuracy: 0.4917
12960/25000 [==============>...............] - ETA: 33s - loss: 7.7920 - accuracy: 0.4918
12992/25000 [==============>...............] - ETA: 33s - loss: 7.7882 - accuracy: 0.4921
13024/25000 [==============>...............] - ETA: 33s - loss: 7.7867 - accuracy: 0.4922
13056/25000 [==============>...............] - ETA: 33s - loss: 7.7876 - accuracy: 0.4921
13088/25000 [==============>...............] - ETA: 33s - loss: 7.7896 - accuracy: 0.4920
13120/25000 [==============>...............] - ETA: 33s - loss: 7.7928 - accuracy: 0.4918
13152/25000 [==============>...............] - ETA: 33s - loss: 7.7937 - accuracy: 0.4917
13184/25000 [==============>...............] - ETA: 33s - loss: 7.7911 - accuracy: 0.4919
13216/25000 [==============>...............] - ETA: 32s - loss: 7.7908 - accuracy: 0.4919
13248/25000 [==============>...............] - ETA: 32s - loss: 7.7928 - accuracy: 0.4918
13280/25000 [==============>...............] - ETA: 32s - loss: 7.7867 - accuracy: 0.4922
13312/25000 [==============>...............] - ETA: 32s - loss: 7.7853 - accuracy: 0.4923
13344/25000 [===============>..............] - ETA: 32s - loss: 7.7781 - accuracy: 0.4927
13376/25000 [===============>..............] - ETA: 32s - loss: 7.7801 - accuracy: 0.4926
13408/25000 [===============>..............] - ETA: 32s - loss: 7.7833 - accuracy: 0.4924
13440/25000 [===============>..............] - ETA: 32s - loss: 7.7841 - accuracy: 0.4923
13472/25000 [===============>..............] - ETA: 32s - loss: 7.7793 - accuracy: 0.4927
13504/25000 [===============>..............] - ETA: 32s - loss: 7.7881 - accuracy: 0.4921
13536/25000 [===============>..............] - ETA: 32s - loss: 7.7878 - accuracy: 0.4921
13568/25000 [===============>..............] - ETA: 31s - loss: 7.7842 - accuracy: 0.4923
13600/25000 [===============>..............] - ETA: 31s - loss: 7.7827 - accuracy: 0.4924
13632/25000 [===============>..............] - ETA: 31s - loss: 7.7836 - accuracy: 0.4924
13664/25000 [===============>..............] - ETA: 31s - loss: 7.7800 - accuracy: 0.4926
13696/25000 [===============>..............] - ETA: 31s - loss: 7.7797 - accuracy: 0.4926
13728/25000 [===============>..............] - ETA: 31s - loss: 7.7783 - accuracy: 0.4927
13760/25000 [===============>..............] - ETA: 31s - loss: 7.7747 - accuracy: 0.4930
13792/25000 [===============>..............] - ETA: 31s - loss: 7.7733 - accuracy: 0.4930
13824/25000 [===============>..............] - ETA: 31s - loss: 7.7709 - accuracy: 0.4932
13856/25000 [===============>..............] - ETA: 31s - loss: 7.7695 - accuracy: 0.4933
13888/25000 [===============>..............] - ETA: 31s - loss: 7.7715 - accuracy: 0.4932
13920/25000 [===============>..............] - ETA: 30s - loss: 7.7702 - accuracy: 0.4932
13952/25000 [===============>..............] - ETA: 30s - loss: 7.7666 - accuracy: 0.4935
13984/25000 [===============>..............] - ETA: 30s - loss: 7.7653 - accuracy: 0.4936
14016/25000 [===============>..............] - ETA: 30s - loss: 7.7684 - accuracy: 0.4934
14048/25000 [===============>..............] - ETA: 30s - loss: 7.7758 - accuracy: 0.4929
14080/25000 [===============>..............] - ETA: 30s - loss: 7.7723 - accuracy: 0.4931
14112/25000 [===============>..............] - ETA: 30s - loss: 7.7688 - accuracy: 0.4933
14144/25000 [===============>..............] - ETA: 30s - loss: 7.7707 - accuracy: 0.4932
14176/25000 [================>.............] - ETA: 30s - loss: 7.7694 - accuracy: 0.4933
14208/25000 [================>.............] - ETA: 30s - loss: 7.7724 - accuracy: 0.4931
14240/25000 [================>.............] - ETA: 30s - loss: 7.7765 - accuracy: 0.4928
14272/25000 [================>.............] - ETA: 29s - loss: 7.7751 - accuracy: 0.4929
14304/25000 [================>.............] - ETA: 29s - loss: 7.7802 - accuracy: 0.4926
14336/25000 [================>.............] - ETA: 29s - loss: 7.7768 - accuracy: 0.4928
14368/25000 [================>.............] - ETA: 29s - loss: 7.7723 - accuracy: 0.4931
14400/25000 [================>.............] - ETA: 29s - loss: 7.7742 - accuracy: 0.4930
14432/25000 [================>.............] - ETA: 29s - loss: 7.7676 - accuracy: 0.4934
14464/25000 [================>.............] - ETA: 29s - loss: 7.7673 - accuracy: 0.4934
14496/25000 [================>.............] - ETA: 29s - loss: 7.7671 - accuracy: 0.4934
14528/25000 [================>.............] - ETA: 29s - loss: 7.7637 - accuracy: 0.4937
14560/25000 [================>.............] - ETA: 29s - loss: 7.7614 - accuracy: 0.4938
14592/25000 [================>.............] - ETA: 29s - loss: 7.7570 - accuracy: 0.4941
14624/25000 [================>.............] - ETA: 28s - loss: 7.7547 - accuracy: 0.4943
14656/25000 [================>.............] - ETA: 28s - loss: 7.7566 - accuracy: 0.4941
14688/25000 [================>.............] - ETA: 28s - loss: 7.7522 - accuracy: 0.4944
14720/25000 [================>.............] - ETA: 28s - loss: 7.7510 - accuracy: 0.4945
14752/25000 [================>.............] - ETA: 28s - loss: 7.7508 - accuracy: 0.4945
14784/25000 [================>.............] - ETA: 28s - loss: 7.7475 - accuracy: 0.4947
14816/25000 [================>.............] - ETA: 28s - loss: 7.7463 - accuracy: 0.4948
14848/25000 [================>.............] - ETA: 28s - loss: 7.7472 - accuracy: 0.4947
14880/25000 [================>.............] - ETA: 28s - loss: 7.7511 - accuracy: 0.4945
14912/25000 [================>.............] - ETA: 28s - loss: 7.7448 - accuracy: 0.4949
14944/25000 [================>.............] - ETA: 28s - loss: 7.7425 - accuracy: 0.4950
14976/25000 [================>.............] - ETA: 27s - loss: 7.7414 - accuracy: 0.4951
15008/25000 [=================>............] - ETA: 27s - loss: 7.7361 - accuracy: 0.4955
15040/25000 [=================>............] - ETA: 27s - loss: 7.7329 - accuracy: 0.4957
15072/25000 [=================>............] - ETA: 27s - loss: 7.7338 - accuracy: 0.4956
15104/25000 [=================>............] - ETA: 27s - loss: 7.7306 - accuracy: 0.4958
15136/25000 [=================>............] - ETA: 27s - loss: 7.7294 - accuracy: 0.4959
15168/25000 [=================>............] - ETA: 27s - loss: 7.7263 - accuracy: 0.4961
15200/25000 [=================>............] - ETA: 27s - loss: 7.7241 - accuracy: 0.4963
15232/25000 [=================>............] - ETA: 27s - loss: 7.7250 - accuracy: 0.4962
15264/25000 [=================>............] - ETA: 27s - loss: 7.7279 - accuracy: 0.4960
15296/25000 [=================>............] - ETA: 27s - loss: 7.7288 - accuracy: 0.4959
15328/25000 [=================>............] - ETA: 26s - loss: 7.7336 - accuracy: 0.4956
15360/25000 [=================>............] - ETA: 26s - loss: 7.7365 - accuracy: 0.4954
15392/25000 [=================>............] - ETA: 26s - loss: 7.7364 - accuracy: 0.4955
15424/25000 [=================>............] - ETA: 26s - loss: 7.7372 - accuracy: 0.4954
15456/25000 [=================>............] - ETA: 26s - loss: 7.7390 - accuracy: 0.4953
15488/25000 [=================>............] - ETA: 26s - loss: 7.7419 - accuracy: 0.4951
15520/25000 [=================>............] - ETA: 26s - loss: 7.7417 - accuracy: 0.4951
15552/25000 [=================>............] - ETA: 26s - loss: 7.7455 - accuracy: 0.4949
15584/25000 [=================>............] - ETA: 26s - loss: 7.7443 - accuracy: 0.4949
15616/25000 [=================>............] - ETA: 26s - loss: 7.7412 - accuracy: 0.4951
15648/25000 [=================>............] - ETA: 26s - loss: 7.7431 - accuracy: 0.4950
15680/25000 [=================>............] - ETA: 25s - loss: 7.7419 - accuracy: 0.4951
15712/25000 [=================>............] - ETA: 25s - loss: 7.7398 - accuracy: 0.4952
15744/25000 [=================>............] - ETA: 25s - loss: 7.7328 - accuracy: 0.4957
15776/25000 [=================>............] - ETA: 25s - loss: 7.7327 - accuracy: 0.4957
15808/25000 [=================>............] - ETA: 25s - loss: 7.7326 - accuracy: 0.4957
15840/25000 [==================>...........] - ETA: 25s - loss: 7.7315 - accuracy: 0.4958
15872/25000 [==================>...........] - ETA: 25s - loss: 7.7294 - accuracy: 0.4959
15904/25000 [==================>...........] - ETA: 25s - loss: 7.7264 - accuracy: 0.4961
15936/25000 [==================>...........] - ETA: 25s - loss: 7.7282 - accuracy: 0.4960
15968/25000 [==================>...........] - ETA: 25s - loss: 7.7252 - accuracy: 0.4962
16000/25000 [==================>...........] - ETA: 25s - loss: 7.7260 - accuracy: 0.4961
16032/25000 [==================>...........] - ETA: 24s - loss: 7.7288 - accuracy: 0.4959
16064/25000 [==================>...........] - ETA: 24s - loss: 7.7296 - accuracy: 0.4959
16096/25000 [==================>...........] - ETA: 24s - loss: 7.7247 - accuracy: 0.4962
16128/25000 [==================>...........] - ETA: 24s - loss: 7.7199 - accuracy: 0.4965
16160/25000 [==================>...........] - ETA: 24s - loss: 7.7207 - accuracy: 0.4965
16192/25000 [==================>...........] - ETA: 24s - loss: 7.7196 - accuracy: 0.4965
16224/25000 [==================>...........] - ETA: 24s - loss: 7.7224 - accuracy: 0.4964
16256/25000 [==================>...........] - ETA: 24s - loss: 7.7223 - accuracy: 0.4964
16288/25000 [==================>...........] - ETA: 24s - loss: 7.7222 - accuracy: 0.4964
16320/25000 [==================>...........] - ETA: 24s - loss: 7.7192 - accuracy: 0.4966
16352/25000 [==================>...........] - ETA: 24s - loss: 7.7201 - accuracy: 0.4965
16384/25000 [==================>...........] - ETA: 23s - loss: 7.7237 - accuracy: 0.4963
16416/25000 [==================>...........] - ETA: 23s - loss: 7.7255 - accuracy: 0.4962
16448/25000 [==================>...........] - ETA: 23s - loss: 7.7235 - accuracy: 0.4963
16480/25000 [==================>...........] - ETA: 23s - loss: 7.7206 - accuracy: 0.4965
16512/25000 [==================>...........] - ETA: 23s - loss: 7.7186 - accuracy: 0.4966
16544/25000 [==================>...........] - ETA: 23s - loss: 7.7185 - accuracy: 0.4966
16576/25000 [==================>...........] - ETA: 23s - loss: 7.7166 - accuracy: 0.4967
16608/25000 [==================>...........] - ETA: 23s - loss: 7.7174 - accuracy: 0.4967
16640/25000 [==================>...........] - ETA: 23s - loss: 7.7191 - accuracy: 0.4966
16672/25000 [===================>..........] - ETA: 23s - loss: 7.7190 - accuracy: 0.4966
16704/25000 [===================>..........] - ETA: 23s - loss: 7.7153 - accuracy: 0.4968
16736/25000 [===================>..........] - ETA: 22s - loss: 7.7161 - accuracy: 0.4968
16768/25000 [===================>..........] - ETA: 22s - loss: 7.7087 - accuracy: 0.4973
16800/25000 [===================>..........] - ETA: 22s - loss: 7.7132 - accuracy: 0.4970
16832/25000 [===================>..........] - ETA: 22s - loss: 7.7140 - accuracy: 0.4969
16864/25000 [===================>..........] - ETA: 22s - loss: 7.7130 - accuracy: 0.4970
16896/25000 [===================>..........] - ETA: 22s - loss: 7.7111 - accuracy: 0.4971
16928/25000 [===================>..........] - ETA: 22s - loss: 7.7083 - accuracy: 0.4973
16960/25000 [===================>..........] - ETA: 22s - loss: 7.7127 - accuracy: 0.4970
16992/25000 [===================>..........] - ETA: 22s - loss: 7.7144 - accuracy: 0.4969
17024/25000 [===================>..........] - ETA: 22s - loss: 7.7171 - accuracy: 0.4967
17056/25000 [===================>..........] - ETA: 22s - loss: 7.7188 - accuracy: 0.4966
17088/25000 [===================>..........] - ETA: 21s - loss: 7.7178 - accuracy: 0.4967
17120/25000 [===================>..........] - ETA: 21s - loss: 7.7177 - accuracy: 0.4967
17152/25000 [===================>..........] - ETA: 21s - loss: 7.7167 - accuracy: 0.4967
17184/25000 [===================>..........] - ETA: 21s - loss: 7.7148 - accuracy: 0.4969
17216/25000 [===================>..........] - ETA: 21s - loss: 7.7147 - accuracy: 0.4969
17248/25000 [===================>..........] - ETA: 21s - loss: 7.7164 - accuracy: 0.4968
17280/25000 [===================>..........] - ETA: 21s - loss: 7.7181 - accuracy: 0.4966
17312/25000 [===================>..........] - ETA: 21s - loss: 7.7189 - accuracy: 0.4966
17344/25000 [===================>..........] - ETA: 21s - loss: 7.7188 - accuracy: 0.4966
17376/25000 [===================>..........] - ETA: 21s - loss: 7.7196 - accuracy: 0.4965
17408/25000 [===================>..........] - ETA: 21s - loss: 7.7195 - accuracy: 0.4966
17440/25000 [===================>..........] - ETA: 20s - loss: 7.7167 - accuracy: 0.4967
17472/25000 [===================>..........] - ETA: 20s - loss: 7.7184 - accuracy: 0.4966
17504/25000 [====================>.........] - ETA: 20s - loss: 7.7166 - accuracy: 0.4967
17536/25000 [====================>.........] - ETA: 20s - loss: 7.7191 - accuracy: 0.4966
17568/25000 [====================>.........] - ETA: 20s - loss: 7.7172 - accuracy: 0.4967
17600/25000 [====================>.........] - ETA: 20s - loss: 7.7145 - accuracy: 0.4969
17632/25000 [====================>.........] - ETA: 20s - loss: 7.7136 - accuracy: 0.4969
17664/25000 [====================>.........] - ETA: 20s - loss: 7.7118 - accuracy: 0.4971
17696/25000 [====================>.........] - ETA: 20s - loss: 7.7108 - accuracy: 0.4971
17728/25000 [====================>.........] - ETA: 20s - loss: 7.7081 - accuracy: 0.4973
17760/25000 [====================>.........] - ETA: 20s - loss: 7.7081 - accuracy: 0.4973
17792/25000 [====================>.........] - ETA: 19s - loss: 7.7063 - accuracy: 0.4974
17824/25000 [====================>.........] - ETA: 19s - loss: 7.7071 - accuracy: 0.4974
17856/25000 [====================>.........] - ETA: 19s - loss: 7.7061 - accuracy: 0.4974
17888/25000 [====================>.........] - ETA: 19s - loss: 7.7078 - accuracy: 0.4973
17920/25000 [====================>.........] - ETA: 19s - loss: 7.7137 - accuracy: 0.4969
17952/25000 [====================>.........] - ETA: 19s - loss: 7.7145 - accuracy: 0.4969
17984/25000 [====================>.........] - ETA: 19s - loss: 7.7161 - accuracy: 0.4968
18016/25000 [====================>.........] - ETA: 19s - loss: 7.7160 - accuracy: 0.4968
18048/25000 [====================>.........] - ETA: 19s - loss: 7.7133 - accuracy: 0.4970
18080/25000 [====================>.........] - ETA: 19s - loss: 7.7133 - accuracy: 0.4970
18112/25000 [====================>.........] - ETA: 19s - loss: 7.7123 - accuracy: 0.4970
18144/25000 [====================>.........] - ETA: 19s - loss: 7.7207 - accuracy: 0.4965
18176/25000 [====================>.........] - ETA: 18s - loss: 7.7181 - accuracy: 0.4966
18208/25000 [====================>.........] - ETA: 18s - loss: 7.7180 - accuracy: 0.4966
18240/25000 [====================>.........] - ETA: 18s - loss: 7.7179 - accuracy: 0.4967
18272/25000 [====================>.........] - ETA: 18s - loss: 7.7178 - accuracy: 0.4967
18304/25000 [====================>.........] - ETA: 18s - loss: 7.7186 - accuracy: 0.4966
18336/25000 [=====================>........] - ETA: 18s - loss: 7.7185 - accuracy: 0.4966
18368/25000 [=====================>........] - ETA: 18s - loss: 7.7175 - accuracy: 0.4967
18400/25000 [=====================>........] - ETA: 18s - loss: 7.7158 - accuracy: 0.4968
18432/25000 [=====================>........] - ETA: 18s - loss: 7.7124 - accuracy: 0.4970
18464/25000 [=====================>........] - ETA: 18s - loss: 7.7140 - accuracy: 0.4969
18496/25000 [=====================>........] - ETA: 18s - loss: 7.7130 - accuracy: 0.4970
18528/25000 [=====================>........] - ETA: 17s - loss: 7.7171 - accuracy: 0.4967
18560/25000 [=====================>........] - ETA: 17s - loss: 7.7145 - accuracy: 0.4969
18592/25000 [=====================>........] - ETA: 17s - loss: 7.7136 - accuracy: 0.4969
18624/25000 [=====================>........] - ETA: 17s - loss: 7.7160 - accuracy: 0.4968
18656/25000 [=====================>........] - ETA: 17s - loss: 7.7159 - accuracy: 0.4968
18688/25000 [=====================>........] - ETA: 17s - loss: 7.7183 - accuracy: 0.4966
18720/25000 [=====================>........] - ETA: 17s - loss: 7.7166 - accuracy: 0.4967
18752/25000 [=====================>........] - ETA: 17s - loss: 7.7173 - accuracy: 0.4967
18784/25000 [=====================>........] - ETA: 17s - loss: 7.7164 - accuracy: 0.4968
18816/25000 [=====================>........] - ETA: 17s - loss: 7.7147 - accuracy: 0.4969
18848/25000 [=====================>........] - ETA: 17s - loss: 7.7154 - accuracy: 0.4968
18880/25000 [=====================>........] - ETA: 16s - loss: 7.7113 - accuracy: 0.4971
18912/25000 [=====================>........] - ETA: 16s - loss: 7.7112 - accuracy: 0.4971
18944/25000 [=====================>........] - ETA: 16s - loss: 7.7111 - accuracy: 0.4971
18976/25000 [=====================>........] - ETA: 16s - loss: 7.7127 - accuracy: 0.4970
19008/25000 [=====================>........] - ETA: 16s - loss: 7.7126 - accuracy: 0.4970
19040/25000 [=====================>........] - ETA: 16s - loss: 7.7133 - accuracy: 0.4970
19072/25000 [=====================>........] - ETA: 16s - loss: 7.7116 - accuracy: 0.4971
19104/25000 [=====================>........] - ETA: 16s - loss: 7.7108 - accuracy: 0.4971
19136/25000 [=====================>........] - ETA: 16s - loss: 7.7139 - accuracy: 0.4969
19168/25000 [======================>.......] - ETA: 16s - loss: 7.7162 - accuracy: 0.4968
19200/25000 [======================>.......] - ETA: 16s - loss: 7.7153 - accuracy: 0.4968
19232/25000 [======================>.......] - ETA: 15s - loss: 7.7168 - accuracy: 0.4967
19264/25000 [======================>.......] - ETA: 15s - loss: 7.7176 - accuracy: 0.4967
19296/25000 [======================>.......] - ETA: 15s - loss: 7.7143 - accuracy: 0.4969
19328/25000 [======================>.......] - ETA: 15s - loss: 7.7150 - accuracy: 0.4968
19360/25000 [======================>.......] - ETA: 15s - loss: 7.7157 - accuracy: 0.4968
19392/25000 [======================>.......] - ETA: 15s - loss: 7.7149 - accuracy: 0.4969
19424/25000 [======================>.......] - ETA: 15s - loss: 7.7164 - accuracy: 0.4968
19456/25000 [======================>.......] - ETA: 15s - loss: 7.7171 - accuracy: 0.4967
19488/25000 [======================>.......] - ETA: 15s - loss: 7.7123 - accuracy: 0.4970
19520/25000 [======================>.......] - ETA: 15s - loss: 7.7130 - accuracy: 0.4970
19552/25000 [======================>.......] - ETA: 15s - loss: 7.7121 - accuracy: 0.4970
19584/25000 [======================>.......] - ETA: 14s - loss: 7.7097 - accuracy: 0.4972
19616/25000 [======================>.......] - ETA: 14s - loss: 7.7104 - accuracy: 0.4971
19648/25000 [======================>.......] - ETA: 14s - loss: 7.7095 - accuracy: 0.4972
19680/25000 [======================>.......] - ETA: 14s - loss: 7.7110 - accuracy: 0.4971
19712/25000 [======================>.......] - ETA: 14s - loss: 7.7094 - accuracy: 0.4972
19744/25000 [======================>.......] - ETA: 14s - loss: 7.7086 - accuracy: 0.4973
19776/25000 [======================>.......] - ETA: 14s - loss: 7.7069 - accuracy: 0.4974
19808/25000 [======================>.......] - ETA: 14s - loss: 7.7069 - accuracy: 0.4974
19840/25000 [======================>.......] - ETA: 14s - loss: 7.7068 - accuracy: 0.4974
19872/25000 [======================>.......] - ETA: 14s - loss: 7.7060 - accuracy: 0.4974
19904/25000 [======================>.......] - ETA: 14s - loss: 7.7059 - accuracy: 0.4974
19936/25000 [======================>.......] - ETA: 14s - loss: 7.7082 - accuracy: 0.4973
19968/25000 [======================>.......] - ETA: 13s - loss: 7.7089 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 13s - loss: 7.7088 - accuracy: 0.4972
20032/25000 [=======================>......] - ETA: 13s - loss: 7.7118 - accuracy: 0.4971
20064/25000 [=======================>......] - ETA: 13s - loss: 7.7117 - accuracy: 0.4971
20096/25000 [=======================>......] - ETA: 13s - loss: 7.7101 - accuracy: 0.4972
20128/25000 [=======================>......] - ETA: 13s - loss: 7.7085 - accuracy: 0.4973
20160/25000 [=======================>......] - ETA: 13s - loss: 7.7085 - accuracy: 0.4973
20192/25000 [=======================>......] - ETA: 13s - loss: 7.7069 - accuracy: 0.4974
20224/25000 [=======================>......] - ETA: 13s - loss: 7.7083 - accuracy: 0.4973
20256/25000 [=======================>......] - ETA: 13s - loss: 7.7060 - accuracy: 0.4974
20288/25000 [=======================>......] - ETA: 13s - loss: 7.7112 - accuracy: 0.4971
20320/25000 [=======================>......] - ETA: 12s - loss: 7.7089 - accuracy: 0.4972
20352/25000 [=======================>......] - ETA: 12s - loss: 7.7028 - accuracy: 0.4976
20384/25000 [=======================>......] - ETA: 12s - loss: 7.7027 - accuracy: 0.4976
20416/25000 [=======================>......] - ETA: 12s - loss: 7.7034 - accuracy: 0.4976
20448/25000 [=======================>......] - ETA: 12s - loss: 7.7034 - accuracy: 0.4976
20480/25000 [=======================>......] - ETA: 12s - loss: 7.7070 - accuracy: 0.4974
20512/25000 [=======================>......] - ETA: 12s - loss: 7.7070 - accuracy: 0.4974
20544/25000 [=======================>......] - ETA: 12s - loss: 7.7054 - accuracy: 0.4975
20576/25000 [=======================>......] - ETA: 12s - loss: 7.7061 - accuracy: 0.4974
20608/25000 [=======================>......] - ETA: 12s - loss: 7.7031 - accuracy: 0.4976
20640/25000 [=======================>......] - ETA: 12s - loss: 7.7030 - accuracy: 0.4976
20672/25000 [=======================>......] - ETA: 11s - loss: 7.7044 - accuracy: 0.4975
20704/25000 [=======================>......] - ETA: 11s - loss: 7.7044 - accuracy: 0.4975
20736/25000 [=======================>......] - ETA: 11s - loss: 7.7043 - accuracy: 0.4975
20768/25000 [=======================>......] - ETA: 11s - loss: 7.7035 - accuracy: 0.4976
20800/25000 [=======================>......] - ETA: 11s - loss: 7.7013 - accuracy: 0.4977
20832/25000 [=======================>......] - ETA: 11s - loss: 7.6968 - accuracy: 0.4980
20864/25000 [========================>.....] - ETA: 11s - loss: 7.6968 - accuracy: 0.4980
20896/25000 [========================>.....] - ETA: 11s - loss: 7.6982 - accuracy: 0.4979
20928/25000 [========================>.....] - ETA: 11s - loss: 7.6989 - accuracy: 0.4979
20960/25000 [========================>.....] - ETA: 11s - loss: 7.6973 - accuracy: 0.4980
20992/25000 [========================>.....] - ETA: 11s - loss: 7.6936 - accuracy: 0.4982
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6965 - accuracy: 0.4980
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6965 - accuracy: 0.4981
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6935 - accuracy: 0.4982
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6949 - accuracy: 0.4982
21152/25000 [========================>.....] - ETA: 10s - loss: 7.6985 - accuracy: 0.4979
21184/25000 [========================>.....] - ETA: 10s - loss: 7.6999 - accuracy: 0.4978
21216/25000 [========================>.....] - ETA: 10s - loss: 7.6999 - accuracy: 0.4978
21248/25000 [========================>.....] - ETA: 10s - loss: 7.6991 - accuracy: 0.4979
21280/25000 [========================>.....] - ETA: 10s - loss: 7.6969 - accuracy: 0.4980
21312/25000 [========================>.....] - ETA: 10s - loss: 7.6976 - accuracy: 0.4980
21344/25000 [========================>.....] - ETA: 10s - loss: 7.6989 - accuracy: 0.4979
21376/25000 [========================>.....] - ETA: 10s - loss: 7.7011 - accuracy: 0.4978
21408/25000 [========================>.....] - ETA: 9s - loss: 7.7046 - accuracy: 0.4975 
21440/25000 [========================>.....] - ETA: 9s - loss: 7.7024 - accuracy: 0.4977
21472/25000 [========================>.....] - ETA: 9s - loss: 7.7052 - accuracy: 0.4975
21504/25000 [========================>.....] - ETA: 9s - loss: 7.7058 - accuracy: 0.4974
21536/25000 [========================>.....] - ETA: 9s - loss: 7.7051 - accuracy: 0.4975
21568/25000 [========================>.....] - ETA: 9s - loss: 7.7029 - accuracy: 0.4976
21600/25000 [========================>.....] - ETA: 9s - loss: 7.7057 - accuracy: 0.4975
21632/25000 [========================>.....] - ETA: 9s - loss: 7.7042 - accuracy: 0.4975
21664/25000 [========================>.....] - ETA: 9s - loss: 7.7063 - accuracy: 0.4974
21696/25000 [=========================>....] - ETA: 9s - loss: 7.7076 - accuracy: 0.4973
21728/25000 [=========================>....] - ETA: 9s - loss: 7.7068 - accuracy: 0.4974
21760/25000 [=========================>....] - ETA: 8s - loss: 7.7068 - accuracy: 0.4974
21792/25000 [=========================>....] - ETA: 8s - loss: 7.7088 - accuracy: 0.4972
21824/25000 [=========================>....] - ETA: 8s - loss: 7.7081 - accuracy: 0.4973
21856/25000 [=========================>....] - ETA: 8s - loss: 7.7052 - accuracy: 0.4975
21888/25000 [=========================>....] - ETA: 8s - loss: 7.7030 - accuracy: 0.4976
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6995 - accuracy: 0.4979
21952/25000 [=========================>....] - ETA: 8s - loss: 7.6987 - accuracy: 0.4979
21984/25000 [=========================>....] - ETA: 8s - loss: 7.6952 - accuracy: 0.4981
22016/25000 [=========================>....] - ETA: 8s - loss: 7.6959 - accuracy: 0.4981
22048/25000 [=========================>....] - ETA: 8s - loss: 7.6930 - accuracy: 0.4983
22080/25000 [=========================>....] - ETA: 8s - loss: 7.6916 - accuracy: 0.4984
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6937 - accuracy: 0.4982
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6922 - accuracy: 0.4983
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6929 - accuracy: 0.4983
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6880 - accuracy: 0.4986
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6880 - accuracy: 0.4986
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6859 - accuracy: 0.4987
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6831 - accuracy: 0.4989
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6845 - accuracy: 0.4988
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6824 - accuracy: 0.4990
22400/25000 [=========================>....] - ETA: 7s - loss: 7.6830 - accuracy: 0.4989
22432/25000 [=========================>....] - ETA: 7s - loss: 7.6837 - accuracy: 0.4989
22464/25000 [=========================>....] - ETA: 7s - loss: 7.6830 - accuracy: 0.4989
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6843 - accuracy: 0.4988
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6850 - accuracy: 0.4988
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6863 - accuracy: 0.4987
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6883 - accuracy: 0.4986
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6863 - accuracy: 0.4987
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6842 - accuracy: 0.4989
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6849 - accuracy: 0.4988
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6848 - accuracy: 0.4988
22752/25000 [==========================>...] - ETA: 6s - loss: 7.6814 - accuracy: 0.4990
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6808 - accuracy: 0.4991
22816/25000 [==========================>...] - ETA: 6s - loss: 7.6814 - accuracy: 0.4990
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6807 - accuracy: 0.4991
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6773 - accuracy: 0.4993
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6760 - accuracy: 0.4994
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6753 - accuracy: 0.4994
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6780 - accuracy: 0.4993
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6740 - accuracy: 0.4995
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6759 - accuracy: 0.4994
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6773 - accuracy: 0.4993
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6786 - accuracy: 0.4992
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6766 - accuracy: 0.4994
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6785 - accuracy: 0.4992
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6759 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6745 - accuracy: 0.4995
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6738 - accuracy: 0.4995
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6738 - accuracy: 0.4995
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6745 - accuracy: 0.4995
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6738 - accuracy: 0.4995
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6763 - accuracy: 0.4994
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6756 - accuracy: 0.4994
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6756 - accuracy: 0.4994
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6768 - accuracy: 0.4993
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6781 - accuracy: 0.4993
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6768 - accuracy: 0.4993
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6755 - accuracy: 0.4994
24192/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24224/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24256/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24288/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24320/25000 [============================>.] - ETA: 1s - loss: 7.6767 - accuracy: 0.4993
24352/25000 [============================>.] - ETA: 1s - loss: 7.6773 - accuracy: 0.4993
24384/25000 [============================>.] - ETA: 1s - loss: 7.6767 - accuracy: 0.4993
24416/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24480/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24544/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24640/25000 [============================>.] - ETA: 0s - loss: 7.6704 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24768/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24832/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 82s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

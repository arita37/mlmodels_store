
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c769f0a4982c13455623165ee1137484d55bac83', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'c769f0a4982c13455623165ee1137484d55bac83', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c769f0a4982c13455623165ee1137484d55bac83

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c769f0a4982c13455623165ee1137484d55bac83

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/c769f0a4982c13455623165ee1137484d55bac83

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
 40%|████      | 2/5 [00:53<01:20, 26.98s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.0889986706600261, 'embedding_size_factor': 0.7471762080129627, 'layers.choice': 0, 'learning_rate': 0.000595490129543915, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.00034076431001620034} and reward: 0.3732
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xb6\xc8\x9d\xeb\xdfN1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\xe8\xde\x148z\x9eX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?C\x83UP\xd4fFX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?6U\x13\x911\xa1\xfcu.' and reward: 0.3732
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xb6\xc8\x9d\xeb\xdfN1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\xe8\xde\x148z\x9eX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?C\x83UP\xd4fFX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?6U\x13\x911\xa1\xfcu.' and reward: 0.3732
 60%|██████    | 3/5 [01:49<01:10, 35.49s/it] 60%|██████    | 3/5 [01:49<01:12, 36.43s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.29634081739676554, 'embedding_size_factor': 1.422476329050423, 'layers.choice': 2, 'learning_rate': 0.00036366365331774617, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 2.242956080691371e-09} and reward: 0.3508
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xf7?y\xcc\x18\xd6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\xc2v\x8a\t\xb0\xf4X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?7\xd5C\x7fl\x1f\xebX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>#DP\x05j+0u.' and reward: 0.3508
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xf7?y\xcc\x18\xd6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\xc2v\x8a\t\xb0\xf4X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?7\xd5C\x7fl\x1f\xebX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>#DP\x05j+0u.' and reward: 0.3508
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 166.00262784957886
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -48.79s of remaining time.
Ensemble size: 76
Ensemble weights: 
[0.65789474 0.19736842 0.14473684]
	0.3898	 = Validation accuracy score
	1.08s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 169.91s ...
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
 2121728/17464789 [==>...........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
17047552/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-23 02:15:17.995067: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 02:15:17.999095: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-05-23 02:15:17.999225: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d9ca09d2d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 02:15:17.999242: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:56 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 3:10 - loss: 8.3854 - accuracy: 0.4531
   96/25000 [..............................] - ETA: 2:32 - loss: 8.4652 - accuracy: 0.4479
  128/25000 [..............................] - ETA: 2:13 - loss: 8.0260 - accuracy: 0.4766
  160/25000 [..............................] - ETA: 2:03 - loss: 8.0500 - accuracy: 0.4750
  192/25000 [..............................] - ETA: 1:56 - loss: 8.2256 - accuracy: 0.4635
  224/25000 [..............................] - ETA: 1:51 - loss: 8.2142 - accuracy: 0.4643
  256/25000 [..............................] - ETA: 1:47 - loss: 7.9661 - accuracy: 0.4805
  288/25000 [..............................] - ETA: 1:44 - loss: 7.6134 - accuracy: 0.5035
  320/25000 [..............................] - ETA: 1:41 - loss: 7.5229 - accuracy: 0.5094
  352/25000 [..............................] - ETA: 1:39 - loss: 7.4924 - accuracy: 0.5114
  384/25000 [..............................] - ETA: 1:37 - loss: 7.4670 - accuracy: 0.5130
  416/25000 [..............................] - ETA: 1:35 - loss: 7.6666 - accuracy: 0.5000
  448/25000 [..............................] - ETA: 1:33 - loss: 7.7351 - accuracy: 0.4955
  480/25000 [..............................] - ETA: 1:32 - loss: 7.7944 - accuracy: 0.4917
  512/25000 [..............................] - ETA: 1:32 - loss: 7.6666 - accuracy: 0.5000
  544/25000 [..............................] - ETA: 1:31 - loss: 7.6102 - accuracy: 0.5037
  576/25000 [..............................] - ETA: 1:30 - loss: 7.6134 - accuracy: 0.5035
  608/25000 [..............................] - ETA: 1:30 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:29 - loss: 7.6187 - accuracy: 0.5031
  672/25000 [..............................] - ETA: 1:28 - loss: 7.7123 - accuracy: 0.4970
  704/25000 [..............................] - ETA: 1:28 - loss: 7.6884 - accuracy: 0.4986
  736/25000 [..............................] - ETA: 1:27 - loss: 7.6875 - accuracy: 0.4986
  768/25000 [..............................] - ETA: 1:26 - loss: 7.6666 - accuracy: 0.5000
  800/25000 [..............................] - ETA: 1:26 - loss: 7.7625 - accuracy: 0.4938
  832/25000 [..............................] - ETA: 1:26 - loss: 7.7956 - accuracy: 0.4916
  864/25000 [>.............................] - ETA: 1:26 - loss: 7.7908 - accuracy: 0.4919
  896/25000 [>.............................] - ETA: 1:25 - loss: 7.7864 - accuracy: 0.4922
  928/25000 [>.............................] - ETA: 1:24 - loss: 7.7988 - accuracy: 0.4914
  960/25000 [>.............................] - ETA: 1:24 - loss: 7.8104 - accuracy: 0.4906
  992/25000 [>.............................] - ETA: 1:23 - loss: 7.7594 - accuracy: 0.4940
 1024/25000 [>.............................] - ETA: 1:23 - loss: 7.7565 - accuracy: 0.4941
 1056/25000 [>.............................] - ETA: 1:22 - loss: 7.7247 - accuracy: 0.4962
 1088/25000 [>.............................] - ETA: 1:22 - loss: 7.7935 - accuracy: 0.4917
 1120/25000 [>.............................] - ETA: 1:22 - loss: 7.7898 - accuracy: 0.4920
 1152/25000 [>.............................] - ETA: 1:21 - loss: 7.7731 - accuracy: 0.4931
 1184/25000 [>.............................] - ETA: 1:21 - loss: 7.7573 - accuracy: 0.4941
 1216/25000 [>.............................] - ETA: 1:21 - loss: 7.7171 - accuracy: 0.4967
 1248/25000 [>.............................] - ETA: 1:21 - loss: 7.6666 - accuracy: 0.5000
 1280/25000 [>.............................] - ETA: 1:20 - loss: 7.6906 - accuracy: 0.4984
 1312/25000 [>.............................] - ETA: 1:20 - loss: 7.6666 - accuracy: 0.5000
 1344/25000 [>.............................] - ETA: 1:20 - loss: 7.7237 - accuracy: 0.4963
 1376/25000 [>.............................] - ETA: 1:20 - loss: 7.7112 - accuracy: 0.4971
 1408/25000 [>.............................] - ETA: 1:20 - loss: 7.6775 - accuracy: 0.4993
 1440/25000 [>.............................] - ETA: 1:19 - loss: 7.6879 - accuracy: 0.4986
 1472/25000 [>.............................] - ETA: 1:19 - loss: 7.6875 - accuracy: 0.4986
 1504/25000 [>.............................] - ETA: 1:19 - loss: 7.6768 - accuracy: 0.4993
 1536/25000 [>.............................] - ETA: 1:19 - loss: 7.7065 - accuracy: 0.4974
 1568/25000 [>.............................] - ETA: 1:19 - loss: 7.7449 - accuracy: 0.4949
 1600/25000 [>.............................] - ETA: 1:18 - loss: 7.7720 - accuracy: 0.4931
 1632/25000 [>.............................] - ETA: 1:18 - loss: 7.7700 - accuracy: 0.4933
 1664/25000 [>.............................] - ETA: 1:18 - loss: 7.7311 - accuracy: 0.4958
 1696/25000 [=>............................] - ETA: 1:18 - loss: 7.7389 - accuracy: 0.4953
 1728/25000 [=>............................] - ETA: 1:17 - loss: 7.7908 - accuracy: 0.4919
 1760/25000 [=>............................] - ETA: 1:17 - loss: 7.8234 - accuracy: 0.4898
 1792/25000 [=>............................] - ETA: 1:17 - loss: 7.7950 - accuracy: 0.4916
 1824/25000 [=>............................] - ETA: 1:17 - loss: 7.7759 - accuracy: 0.4929
 1856/25000 [=>............................] - ETA: 1:17 - loss: 7.7823 - accuracy: 0.4925
 1888/25000 [=>............................] - ETA: 1:17 - loss: 7.7722 - accuracy: 0.4931
 1920/25000 [=>............................] - ETA: 1:16 - loss: 7.7944 - accuracy: 0.4917
 1952/25000 [=>............................] - ETA: 1:16 - loss: 7.7844 - accuracy: 0.4923
 1984/25000 [=>............................] - ETA: 1:16 - loss: 7.7825 - accuracy: 0.4924
 2016/25000 [=>............................] - ETA: 1:16 - loss: 7.7655 - accuracy: 0.4936
 2048/25000 [=>............................] - ETA: 1:16 - loss: 7.7340 - accuracy: 0.4956
 2080/25000 [=>............................] - ETA: 1:16 - loss: 7.6814 - accuracy: 0.4990
 2112/25000 [=>............................] - ETA: 1:16 - loss: 7.6884 - accuracy: 0.4986
 2144/25000 [=>............................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 2176/25000 [=>............................] - ETA: 1:15 - loss: 7.6878 - accuracy: 0.4986
 2208/25000 [=>............................] - ETA: 1:15 - loss: 7.6527 - accuracy: 0.5009
 2240/25000 [=>............................] - ETA: 1:15 - loss: 7.6461 - accuracy: 0.5013
 2272/25000 [=>............................] - ETA: 1:15 - loss: 7.6464 - accuracy: 0.5013
 2304/25000 [=>............................] - ETA: 1:15 - loss: 7.6400 - accuracy: 0.5017
 2336/25000 [=>............................] - ETA: 1:15 - loss: 7.6535 - accuracy: 0.5009
 2368/25000 [=>............................] - ETA: 1:15 - loss: 7.6731 - accuracy: 0.4996
 2400/25000 [=>............................] - ETA: 1:15 - loss: 7.6602 - accuracy: 0.5004
 2432/25000 [=>............................] - ETA: 1:15 - loss: 7.6792 - accuracy: 0.4992
 2464/25000 [=>............................] - ETA: 1:14 - loss: 7.6977 - accuracy: 0.4980
 2496/25000 [=>............................] - ETA: 1:14 - loss: 7.6912 - accuracy: 0.4984
 2528/25000 [==>...........................] - ETA: 1:14 - loss: 7.7030 - accuracy: 0.4976
 2560/25000 [==>...........................] - ETA: 1:14 - loss: 7.6846 - accuracy: 0.4988
 2592/25000 [==>...........................] - ETA: 1:14 - loss: 7.7021 - accuracy: 0.4977
 2624/25000 [==>...........................] - ETA: 1:13 - loss: 7.6841 - accuracy: 0.4989
 2656/25000 [==>...........................] - ETA: 1:13 - loss: 7.6782 - accuracy: 0.4992
 2688/25000 [==>...........................] - ETA: 1:13 - loss: 7.6723 - accuracy: 0.4996
 2720/25000 [==>...........................] - ETA: 1:13 - loss: 7.6553 - accuracy: 0.5007
 2752/25000 [==>...........................] - ETA: 1:13 - loss: 7.6722 - accuracy: 0.4996
 2784/25000 [==>...........................] - ETA: 1:13 - loss: 7.6666 - accuracy: 0.5000
 2816/25000 [==>...........................] - ETA: 1:13 - loss: 7.6503 - accuracy: 0.5011
 2848/25000 [==>...........................] - ETA: 1:13 - loss: 7.6343 - accuracy: 0.5021
 2880/25000 [==>...........................] - ETA: 1:13 - loss: 7.6347 - accuracy: 0.5021
 2912/25000 [==>...........................] - ETA: 1:12 - loss: 7.6245 - accuracy: 0.5027
 2944/25000 [==>...........................] - ETA: 1:12 - loss: 7.6197 - accuracy: 0.5031
 2976/25000 [==>...........................] - ETA: 1:12 - loss: 7.6048 - accuracy: 0.5040
 3008/25000 [==>...........................] - ETA: 1:12 - loss: 7.6258 - accuracy: 0.5027
 3040/25000 [==>...........................] - ETA: 1:12 - loss: 7.6464 - accuracy: 0.5013
 3072/25000 [==>...........................] - ETA: 1:12 - loss: 7.6467 - accuracy: 0.5013
 3104/25000 [==>...........................] - ETA: 1:12 - loss: 7.6370 - accuracy: 0.5019
 3136/25000 [==>...........................] - ETA: 1:12 - loss: 7.6520 - accuracy: 0.5010
 3168/25000 [==>...........................] - ETA: 1:11 - loss: 7.6327 - accuracy: 0.5022
 3200/25000 [==>...........................] - ETA: 1:11 - loss: 7.6522 - accuracy: 0.5009
 3232/25000 [==>...........................] - ETA: 1:11 - loss: 7.6429 - accuracy: 0.5015
 3264/25000 [==>...........................] - ETA: 1:11 - loss: 7.6290 - accuracy: 0.5025
 3296/25000 [==>...........................] - ETA: 1:11 - loss: 7.6294 - accuracy: 0.5024
 3328/25000 [==>...........................] - ETA: 1:11 - loss: 7.6298 - accuracy: 0.5024
 3360/25000 [===>..........................] - ETA: 1:11 - loss: 7.6392 - accuracy: 0.5018
 3392/25000 [===>..........................] - ETA: 1:11 - loss: 7.6395 - accuracy: 0.5018
 3424/25000 [===>..........................] - ETA: 1:11 - loss: 7.6308 - accuracy: 0.5023
 3456/25000 [===>..........................] - ETA: 1:11 - loss: 7.6444 - accuracy: 0.5014
 3488/25000 [===>..........................] - ETA: 1:10 - loss: 7.6710 - accuracy: 0.4997
 3520/25000 [===>..........................] - ETA: 1:10 - loss: 7.6623 - accuracy: 0.5003
 3552/25000 [===>..........................] - ETA: 1:10 - loss: 7.6753 - accuracy: 0.4994
 3584/25000 [===>..........................] - ETA: 1:10 - loss: 7.6880 - accuracy: 0.4986
 3616/25000 [===>..........................] - ETA: 1:10 - loss: 7.6836 - accuracy: 0.4989
 3648/25000 [===>..........................] - ETA: 1:10 - loss: 7.6708 - accuracy: 0.4997
 3680/25000 [===>..........................] - ETA: 1:10 - loss: 7.6583 - accuracy: 0.5005
 3712/25000 [===>..........................] - ETA: 1:10 - loss: 7.6418 - accuracy: 0.5016
 3744/25000 [===>..........................] - ETA: 1:10 - loss: 7.6380 - accuracy: 0.5019
 3776/25000 [===>..........................] - ETA: 1:10 - loss: 7.6463 - accuracy: 0.5013
 3808/25000 [===>..........................] - ETA: 1:09 - loss: 7.6465 - accuracy: 0.5013
 3840/25000 [===>..........................] - ETA: 1:09 - loss: 7.6586 - accuracy: 0.5005
 3872/25000 [===>..........................] - ETA: 1:09 - loss: 7.6349 - accuracy: 0.5021
 3904/25000 [===>..........................] - ETA: 1:09 - loss: 7.6195 - accuracy: 0.5031
 3936/25000 [===>..........................] - ETA: 1:09 - loss: 7.6277 - accuracy: 0.5025
 3968/25000 [===>..........................] - ETA: 1:09 - loss: 7.6396 - accuracy: 0.5018
 4000/25000 [===>..........................] - ETA: 1:09 - loss: 7.6398 - accuracy: 0.5017
 4032/25000 [===>..........................] - ETA: 1:09 - loss: 7.6476 - accuracy: 0.5012
 4064/25000 [===>..........................] - ETA: 1:08 - loss: 7.6440 - accuracy: 0.5015
 4096/25000 [===>..........................] - ETA: 1:08 - loss: 7.6479 - accuracy: 0.5012
 4128/25000 [===>..........................] - ETA: 1:08 - loss: 7.6480 - accuracy: 0.5012
 4160/25000 [===>..........................] - ETA: 1:08 - loss: 7.6629 - accuracy: 0.5002
 4192/25000 [====>.........................] - ETA: 1:08 - loss: 7.6886 - accuracy: 0.4986
 4224/25000 [====>.........................] - ETA: 1:08 - loss: 7.6848 - accuracy: 0.4988
 4256/25000 [====>.........................] - ETA: 1:07 - loss: 7.6810 - accuracy: 0.4991
 4288/25000 [====>.........................] - ETA: 1:07 - loss: 7.6809 - accuracy: 0.4991
 4320/25000 [====>.........................] - ETA: 1:07 - loss: 7.6879 - accuracy: 0.4986
 4352/25000 [====>.........................] - ETA: 1:07 - loss: 7.6878 - accuracy: 0.4986
 4384/25000 [====>.........................] - ETA: 1:07 - loss: 7.7086 - accuracy: 0.4973
 4416/25000 [====>.........................] - ETA: 1:07 - loss: 7.6944 - accuracy: 0.4982
 4448/25000 [====>.........................] - ETA: 1:07 - loss: 7.6839 - accuracy: 0.4989
 4480/25000 [====>.........................] - ETA: 1:06 - loss: 7.6735 - accuracy: 0.4996
 4512/25000 [====>.........................] - ETA: 1:06 - loss: 7.6734 - accuracy: 0.4996
 4544/25000 [====>.........................] - ETA: 1:06 - loss: 7.6734 - accuracy: 0.4996
 4576/25000 [====>.........................] - ETA: 1:06 - loss: 7.6599 - accuracy: 0.5004
 4608/25000 [====>.........................] - ETA: 1:06 - loss: 7.6733 - accuracy: 0.4996
 4640/25000 [====>.........................] - ETA: 1:06 - loss: 7.6831 - accuracy: 0.4989
 4672/25000 [====>.........................] - ETA: 1:06 - loss: 7.6699 - accuracy: 0.4998
 4704/25000 [====>.........................] - ETA: 1:06 - loss: 7.6438 - accuracy: 0.5015
 4736/25000 [====>.........................] - ETA: 1:06 - loss: 7.6472 - accuracy: 0.5013
 4768/25000 [====>.........................] - ETA: 1:05 - loss: 7.6505 - accuracy: 0.5010
 4800/25000 [====>.........................] - ETA: 1:05 - loss: 7.6506 - accuracy: 0.5010
 4832/25000 [====>.........................] - ETA: 1:05 - loss: 7.6444 - accuracy: 0.5014
 4864/25000 [====>.........................] - ETA: 1:05 - loss: 7.6509 - accuracy: 0.5010
 4896/25000 [====>.........................] - ETA: 1:05 - loss: 7.6635 - accuracy: 0.5002
 4928/25000 [====>.........................] - ETA: 1:05 - loss: 7.6760 - accuracy: 0.4994
 4960/25000 [====>.........................] - ETA: 1:05 - loss: 7.6821 - accuracy: 0.4990
 4992/25000 [====>.........................] - ETA: 1:05 - loss: 7.6881 - accuracy: 0.4986
 5024/25000 [=====>........................] - ETA: 1:05 - loss: 7.6941 - accuracy: 0.4982
 5056/25000 [=====>........................] - ETA: 1:04 - loss: 7.6909 - accuracy: 0.4984
 5088/25000 [=====>........................] - ETA: 1:04 - loss: 7.6817 - accuracy: 0.4990
 5120/25000 [=====>........................] - ETA: 1:04 - loss: 7.6876 - accuracy: 0.4986
 5152/25000 [=====>........................] - ETA: 1:04 - loss: 7.6726 - accuracy: 0.4996
 5184/25000 [=====>........................] - ETA: 1:04 - loss: 7.6725 - accuracy: 0.4996
 5216/25000 [=====>........................] - ETA: 1:04 - loss: 7.6872 - accuracy: 0.4987
 5248/25000 [=====>........................] - ETA: 1:04 - loss: 7.6841 - accuracy: 0.4989
 5280/25000 [=====>........................] - ETA: 1:04 - loss: 7.6782 - accuracy: 0.4992
 5312/25000 [=====>........................] - ETA: 1:04 - loss: 7.6724 - accuracy: 0.4996
 5344/25000 [=====>........................] - ETA: 1:04 - loss: 7.6638 - accuracy: 0.5002
 5376/25000 [=====>........................] - ETA: 1:03 - loss: 7.6638 - accuracy: 0.5002
 5408/25000 [=====>........................] - ETA: 1:03 - loss: 7.6638 - accuracy: 0.5002
 5440/25000 [=====>........................] - ETA: 1:03 - loss: 7.6694 - accuracy: 0.4998
 5472/25000 [=====>........................] - ETA: 1:03 - loss: 7.6694 - accuracy: 0.4998
 5504/25000 [=====>........................] - ETA: 1:03 - loss: 7.6833 - accuracy: 0.4989
 5536/25000 [=====>........................] - ETA: 1:03 - loss: 7.6915 - accuracy: 0.4984
 5568/25000 [=====>........................] - ETA: 1:03 - loss: 7.6859 - accuracy: 0.4987
 5600/25000 [=====>........................] - ETA: 1:03 - loss: 7.6748 - accuracy: 0.4995
 5632/25000 [=====>........................] - ETA: 1:03 - loss: 7.6830 - accuracy: 0.4989
 5664/25000 [=====>........................] - ETA: 1:02 - loss: 7.6802 - accuracy: 0.4991
 5696/25000 [=====>........................] - ETA: 1:02 - loss: 7.6747 - accuracy: 0.4995
 5728/25000 [=====>........................] - ETA: 1:02 - loss: 7.6747 - accuracy: 0.4995
 5760/25000 [=====>........................] - ETA: 1:02 - loss: 7.6666 - accuracy: 0.5000
 5792/25000 [=====>........................] - ETA: 1:02 - loss: 7.6746 - accuracy: 0.4995
 5824/25000 [=====>........................] - ETA: 1:02 - loss: 7.6798 - accuracy: 0.4991
 5856/25000 [======>.......................] - ETA: 1:02 - loss: 7.6745 - accuracy: 0.4995
 5888/25000 [======>.......................] - ETA: 1:02 - loss: 7.6718 - accuracy: 0.4997
 5920/25000 [======>.......................] - ETA: 1:02 - loss: 7.6640 - accuracy: 0.5002
 5952/25000 [======>.......................] - ETA: 1:01 - loss: 7.6718 - accuracy: 0.4997
 5984/25000 [======>.......................] - ETA: 1:01 - loss: 7.6769 - accuracy: 0.4993
 6016/25000 [======>.......................] - ETA: 1:01 - loss: 7.6717 - accuracy: 0.4997
 6048/25000 [======>.......................] - ETA: 1:01 - loss: 7.6742 - accuracy: 0.4995
 6080/25000 [======>.......................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 6112/25000 [======>.......................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 6144/25000 [======>.......................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 6176/25000 [======>.......................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 6208/25000 [======>.......................] - ETA: 1:00 - loss: 7.6716 - accuracy: 0.4997
 6240/25000 [======>.......................] - ETA: 1:00 - loss: 7.6715 - accuracy: 0.4997
 6272/25000 [======>.......................] - ETA: 1:00 - loss: 7.6593 - accuracy: 0.5005
 6304/25000 [======>.......................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 6336/25000 [======>.......................] - ETA: 1:00 - loss: 7.6618 - accuracy: 0.5003
 6368/25000 [======>.......................] - ETA: 1:00 - loss: 7.6642 - accuracy: 0.5002
 6400/25000 [======>.......................] - ETA: 1:00 - loss: 7.6618 - accuracy: 0.5003
 6432/25000 [======>.......................] - ETA: 1:00 - loss: 7.6595 - accuracy: 0.5005
 6464/25000 [======>.......................] - ETA: 1:00 - loss: 7.6619 - accuracy: 0.5003
 6496/25000 [======>.......................] - ETA: 1:00 - loss: 7.6690 - accuracy: 0.4998
 6528/25000 [======>.......................] - ETA: 59s - loss: 7.6760 - accuracy: 0.4994 
 6560/25000 [======>.......................] - ETA: 59s - loss: 7.6783 - accuracy: 0.4992
 6592/25000 [======>.......................] - ETA: 59s - loss: 7.6713 - accuracy: 0.4997
 6624/25000 [======>.......................] - ETA: 59s - loss: 7.6712 - accuracy: 0.4997
 6656/25000 [======>.......................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 6688/25000 [=======>......................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 6720/25000 [=======>......................] - ETA: 59s - loss: 7.6735 - accuracy: 0.4996
 6752/25000 [=======>......................] - ETA: 59s - loss: 7.6734 - accuracy: 0.4996
 6784/25000 [=======>......................] - ETA: 59s - loss: 7.6870 - accuracy: 0.4987
 6816/25000 [=======>......................] - ETA: 58s - loss: 7.6824 - accuracy: 0.4990
 6848/25000 [=======>......................] - ETA: 58s - loss: 7.6823 - accuracy: 0.4990
 6880/25000 [=======>......................] - ETA: 58s - loss: 7.6822 - accuracy: 0.4990
 6912/25000 [=======>......................] - ETA: 58s - loss: 7.6799 - accuracy: 0.4991
 6944/25000 [=======>......................] - ETA: 58s - loss: 7.6843 - accuracy: 0.4988
 6976/25000 [=======>......................] - ETA: 58s - loss: 7.6886 - accuracy: 0.4986
 7008/25000 [=======>......................] - ETA: 58s - loss: 7.6819 - accuracy: 0.4990
 7040/25000 [=======>......................] - ETA: 58s - loss: 7.6840 - accuracy: 0.4989
 7072/25000 [=======>......................] - ETA: 58s - loss: 7.6688 - accuracy: 0.4999
 7104/25000 [=======>......................] - ETA: 57s - loss: 7.6623 - accuracy: 0.5003
 7136/25000 [=======>......................] - ETA: 57s - loss: 7.6645 - accuracy: 0.5001
 7168/25000 [=======>......................] - ETA: 57s - loss: 7.6709 - accuracy: 0.4997
 7200/25000 [=======>......................] - ETA: 57s - loss: 7.6687 - accuracy: 0.4999
 7232/25000 [=======>......................] - ETA: 57s - loss: 7.6857 - accuracy: 0.4988
 7264/25000 [=======>......................] - ETA: 57s - loss: 7.6898 - accuracy: 0.4985
 7296/25000 [=======>......................] - ETA: 57s - loss: 7.6897 - accuracy: 0.4985
 7328/25000 [=======>......................] - ETA: 57s - loss: 7.6813 - accuracy: 0.4990
 7360/25000 [=======>......................] - ETA: 57s - loss: 7.6791 - accuracy: 0.4992
 7392/25000 [=======>......................] - ETA: 56s - loss: 7.6811 - accuracy: 0.4991
 7424/25000 [=======>......................] - ETA: 56s - loss: 7.6790 - accuracy: 0.4992
 7456/25000 [=======>......................] - ETA: 56s - loss: 7.6934 - accuracy: 0.4983
 7488/25000 [=======>......................] - ETA: 56s - loss: 7.7076 - accuracy: 0.4973
 7520/25000 [========>.....................] - ETA: 56s - loss: 7.7115 - accuracy: 0.4971
 7552/25000 [========>.....................] - ETA: 56s - loss: 7.7052 - accuracy: 0.4975
 7584/25000 [========>.....................] - ETA: 56s - loss: 7.7050 - accuracy: 0.4975
 7616/25000 [========>.....................] - ETA: 56s - loss: 7.7069 - accuracy: 0.4974
 7648/25000 [========>.....................] - ETA: 56s - loss: 7.7127 - accuracy: 0.4970
 7680/25000 [========>.....................] - ETA: 56s - loss: 7.7105 - accuracy: 0.4971
 7712/25000 [========>.....................] - ETA: 55s - loss: 7.7064 - accuracy: 0.4974
 7744/25000 [========>.....................] - ETA: 55s - loss: 7.7161 - accuracy: 0.4968
 7776/25000 [========>.....................] - ETA: 55s - loss: 7.7159 - accuracy: 0.4968
 7808/25000 [========>.....................] - ETA: 55s - loss: 7.7157 - accuracy: 0.4968
 7840/25000 [========>.....................] - ETA: 55s - loss: 7.7096 - accuracy: 0.4972
 7872/25000 [========>.....................] - ETA: 55s - loss: 7.7134 - accuracy: 0.4970
 7904/25000 [========>.....................] - ETA: 55s - loss: 7.7171 - accuracy: 0.4967
 7936/25000 [========>.....................] - ETA: 55s - loss: 7.7091 - accuracy: 0.4972
 7968/25000 [========>.....................] - ETA: 55s - loss: 7.7070 - accuracy: 0.4974
 8000/25000 [========>.....................] - ETA: 54s - loss: 7.7107 - accuracy: 0.4971
 8032/25000 [========>.....................] - ETA: 54s - loss: 7.7105 - accuracy: 0.4971
 8064/25000 [========>.....................] - ETA: 54s - loss: 7.7065 - accuracy: 0.4974
 8096/25000 [========>.....................] - ETA: 54s - loss: 7.6988 - accuracy: 0.4979
 8128/25000 [========>.....................] - ETA: 54s - loss: 7.6893 - accuracy: 0.4985
 8160/25000 [========>.....................] - ETA: 54s - loss: 7.6835 - accuracy: 0.4989
 8192/25000 [========>.....................] - ETA: 54s - loss: 7.6816 - accuracy: 0.4990
 8224/25000 [========>.....................] - ETA: 54s - loss: 7.6871 - accuracy: 0.4987
 8256/25000 [========>.....................] - ETA: 54s - loss: 7.6889 - accuracy: 0.4985
 8288/25000 [========>.....................] - ETA: 54s - loss: 7.6944 - accuracy: 0.4982
 8320/25000 [========>.....................] - ETA: 53s - loss: 7.6943 - accuracy: 0.4982
 8352/25000 [=========>....................] - ETA: 53s - loss: 7.6960 - accuracy: 0.4981
 8384/25000 [=========>....................] - ETA: 53s - loss: 7.6922 - accuracy: 0.4983
 8416/25000 [=========>....................] - ETA: 53s - loss: 7.6903 - accuracy: 0.4985
 8448/25000 [=========>....................] - ETA: 53s - loss: 7.6975 - accuracy: 0.4980
 8480/25000 [=========>....................] - ETA: 53s - loss: 7.6974 - accuracy: 0.4980
 8512/25000 [=========>....................] - ETA: 53s - loss: 7.6954 - accuracy: 0.4981
 8544/25000 [=========>....................] - ETA: 53s - loss: 7.7043 - accuracy: 0.4975
 8576/25000 [=========>....................] - ETA: 53s - loss: 7.7024 - accuracy: 0.4977
 8608/25000 [=========>....................] - ETA: 53s - loss: 7.7005 - accuracy: 0.4978
 8640/25000 [=========>....................] - ETA: 52s - loss: 7.7092 - accuracy: 0.4972
 8672/25000 [=========>....................] - ETA: 52s - loss: 7.7126 - accuracy: 0.4970
 8704/25000 [=========>....................] - ETA: 52s - loss: 7.7195 - accuracy: 0.4966
 8736/25000 [=========>....................] - ETA: 52s - loss: 7.7210 - accuracy: 0.4965
 8768/25000 [=========>....................] - ETA: 52s - loss: 7.7278 - accuracy: 0.4960
 8800/25000 [=========>....................] - ETA: 52s - loss: 7.7259 - accuracy: 0.4961
 8832/25000 [=========>....................] - ETA: 52s - loss: 7.7274 - accuracy: 0.4960
 8864/25000 [=========>....................] - ETA: 52s - loss: 7.7272 - accuracy: 0.4961
 8896/25000 [=========>....................] - ETA: 52s - loss: 7.7304 - accuracy: 0.4958
 8928/25000 [=========>....................] - ETA: 51s - loss: 7.7319 - accuracy: 0.4957
 8960/25000 [=========>....................] - ETA: 51s - loss: 7.7265 - accuracy: 0.4961
 8992/25000 [=========>....................] - ETA: 51s - loss: 7.7280 - accuracy: 0.4960
 9024/25000 [=========>....................] - ETA: 51s - loss: 7.7278 - accuracy: 0.4960
 9056/25000 [=========>....................] - ETA: 51s - loss: 7.7343 - accuracy: 0.4956
 9088/25000 [=========>....................] - ETA: 51s - loss: 7.7274 - accuracy: 0.4960
 9120/25000 [=========>....................] - ETA: 51s - loss: 7.7255 - accuracy: 0.4962
 9152/25000 [=========>....................] - ETA: 51s - loss: 7.7269 - accuracy: 0.4961
 9184/25000 [==========>...................] - ETA: 51s - loss: 7.7284 - accuracy: 0.4960
 9216/25000 [==========>...................] - ETA: 51s - loss: 7.7232 - accuracy: 0.4963
 9248/25000 [==========>...................] - ETA: 50s - loss: 7.7230 - accuracy: 0.4963
 9280/25000 [==========>...................] - ETA: 50s - loss: 7.7228 - accuracy: 0.4963
 9312/25000 [==========>...................] - ETA: 50s - loss: 7.7243 - accuracy: 0.4962
 9344/25000 [==========>...................] - ETA: 50s - loss: 7.7241 - accuracy: 0.4963
 9376/25000 [==========>...................] - ETA: 50s - loss: 7.7206 - accuracy: 0.4965
 9408/25000 [==========>...................] - ETA: 50s - loss: 7.7220 - accuracy: 0.4964
 9440/25000 [==========>...................] - ETA: 50s - loss: 7.7202 - accuracy: 0.4965
 9472/25000 [==========>...................] - ETA: 50s - loss: 7.7217 - accuracy: 0.4964
 9504/25000 [==========>...................] - ETA: 50s - loss: 7.7247 - accuracy: 0.4962
 9536/25000 [==========>...................] - ETA: 50s - loss: 7.7197 - accuracy: 0.4965
 9568/25000 [==========>...................] - ETA: 49s - loss: 7.7195 - accuracy: 0.4966
 9600/25000 [==========>...................] - ETA: 49s - loss: 7.7193 - accuracy: 0.4966
 9632/25000 [==========>...................] - ETA: 49s - loss: 7.7223 - accuracy: 0.4964
 9664/25000 [==========>...................] - ETA: 49s - loss: 7.7253 - accuracy: 0.4962
 9696/25000 [==========>...................] - ETA: 49s - loss: 7.7315 - accuracy: 0.4958
 9728/25000 [==========>...................] - ETA: 49s - loss: 7.7297 - accuracy: 0.4959
 9760/25000 [==========>...................] - ETA: 49s - loss: 7.7326 - accuracy: 0.4957
 9792/25000 [==========>...................] - ETA: 49s - loss: 7.7324 - accuracy: 0.4957
 9824/25000 [==========>...................] - ETA: 49s - loss: 7.7322 - accuracy: 0.4957
 9856/25000 [==========>...................] - ETA: 48s - loss: 7.7335 - accuracy: 0.4956
 9888/25000 [==========>...................] - ETA: 48s - loss: 7.7364 - accuracy: 0.4954
 9920/25000 [==========>...................] - ETA: 48s - loss: 7.7362 - accuracy: 0.4955
 9952/25000 [==========>...................] - ETA: 48s - loss: 7.7344 - accuracy: 0.4956
 9984/25000 [==========>...................] - ETA: 48s - loss: 7.7342 - accuracy: 0.4956
10016/25000 [===========>..................] - ETA: 48s - loss: 7.7279 - accuracy: 0.4960
10048/25000 [===========>..................] - ETA: 48s - loss: 7.7292 - accuracy: 0.4959
10080/25000 [===========>..................] - ETA: 48s - loss: 7.7290 - accuracy: 0.4959
10112/25000 [===========>..................] - ETA: 48s - loss: 7.7349 - accuracy: 0.4955
10144/25000 [===========>..................] - ETA: 48s - loss: 7.7286 - accuracy: 0.4960
10176/25000 [===========>..................] - ETA: 47s - loss: 7.7269 - accuracy: 0.4961
10208/25000 [===========>..................] - ETA: 47s - loss: 7.7252 - accuracy: 0.4962
10240/25000 [===========>..................] - ETA: 47s - loss: 7.7190 - accuracy: 0.4966
10272/25000 [===========>..................] - ETA: 47s - loss: 7.7174 - accuracy: 0.4967
10304/25000 [===========>..................] - ETA: 47s - loss: 7.7202 - accuracy: 0.4965
10336/25000 [===========>..................] - ETA: 47s - loss: 7.7230 - accuracy: 0.4963
10368/25000 [===========>..................] - ETA: 47s - loss: 7.7184 - accuracy: 0.4966
10400/25000 [===========>..................] - ETA: 47s - loss: 7.7138 - accuracy: 0.4969
10432/25000 [===========>..................] - ETA: 46s - loss: 7.7151 - accuracy: 0.4968
10464/25000 [===========>..................] - ETA: 46s - loss: 7.7091 - accuracy: 0.4972
10496/25000 [===========>..................] - ETA: 46s - loss: 7.7104 - accuracy: 0.4971
10528/25000 [===========>..................] - ETA: 46s - loss: 7.7074 - accuracy: 0.4973
10560/25000 [===========>..................] - ETA: 46s - loss: 7.7087 - accuracy: 0.4973
10592/25000 [===========>..................] - ETA: 46s - loss: 7.7086 - accuracy: 0.4973
10624/25000 [===========>..................] - ETA: 46s - loss: 7.7056 - accuracy: 0.4975
10656/25000 [===========>..................] - ETA: 46s - loss: 7.7083 - accuracy: 0.4973
10688/25000 [===========>..................] - ETA: 46s - loss: 7.7068 - accuracy: 0.4974
10720/25000 [===========>..................] - ETA: 46s - loss: 7.7052 - accuracy: 0.4975
10752/25000 [===========>..................] - ETA: 45s - loss: 7.7094 - accuracy: 0.4972
10784/25000 [===========>..................] - ETA: 45s - loss: 7.7079 - accuracy: 0.4973
10816/25000 [===========>..................] - ETA: 45s - loss: 7.7049 - accuracy: 0.4975
10848/25000 [============>.................] - ETA: 45s - loss: 7.7048 - accuracy: 0.4975
10880/25000 [============>.................] - ETA: 45s - loss: 7.7033 - accuracy: 0.4976
10912/25000 [============>.................] - ETA: 45s - loss: 7.7017 - accuracy: 0.4977
10944/25000 [============>.................] - ETA: 45s - loss: 7.6988 - accuracy: 0.4979
10976/25000 [============>.................] - ETA: 45s - loss: 7.6946 - accuracy: 0.4982
11008/25000 [============>.................] - ETA: 45s - loss: 7.6931 - accuracy: 0.4983
11040/25000 [============>.................] - ETA: 45s - loss: 7.6888 - accuracy: 0.4986
11072/25000 [============>.................] - ETA: 44s - loss: 7.6874 - accuracy: 0.4986
11104/25000 [============>.................] - ETA: 44s - loss: 7.6901 - accuracy: 0.4985
11136/25000 [============>.................] - ETA: 44s - loss: 7.6900 - accuracy: 0.4985
11168/25000 [============>.................] - ETA: 44s - loss: 7.6900 - accuracy: 0.4985
11200/25000 [============>.................] - ETA: 44s - loss: 7.6913 - accuracy: 0.4984
11232/25000 [============>.................] - ETA: 44s - loss: 7.6953 - accuracy: 0.4981
11264/25000 [============>.................] - ETA: 44s - loss: 7.6993 - accuracy: 0.4979
11296/25000 [============>.................] - ETA: 44s - loss: 7.6911 - accuracy: 0.4984
11328/25000 [============>.................] - ETA: 44s - loss: 7.6883 - accuracy: 0.4986
11360/25000 [============>.................] - ETA: 43s - loss: 7.6801 - accuracy: 0.4991
11392/25000 [============>.................] - ETA: 43s - loss: 7.6814 - accuracy: 0.4990
11424/25000 [============>.................] - ETA: 43s - loss: 7.6827 - accuracy: 0.4989
11456/25000 [============>.................] - ETA: 43s - loss: 7.6840 - accuracy: 0.4989
11488/25000 [============>.................] - ETA: 43s - loss: 7.6853 - accuracy: 0.4988
11520/25000 [============>.................] - ETA: 43s - loss: 7.6892 - accuracy: 0.4985
11552/25000 [============>.................] - ETA: 43s - loss: 7.6905 - accuracy: 0.4984
11584/25000 [============>.................] - ETA: 43s - loss: 7.6878 - accuracy: 0.4986
11616/25000 [============>.................] - ETA: 43s - loss: 7.6904 - accuracy: 0.4985
11648/25000 [============>.................] - ETA: 43s - loss: 7.6890 - accuracy: 0.4985
11680/25000 [=============>................] - ETA: 42s - loss: 7.6850 - accuracy: 0.4988
11712/25000 [=============>................] - ETA: 42s - loss: 7.6823 - accuracy: 0.4990
11744/25000 [=============>................] - ETA: 42s - loss: 7.6758 - accuracy: 0.4994
11776/25000 [=============>................] - ETA: 42s - loss: 7.6705 - accuracy: 0.4997
11808/25000 [=============>................] - ETA: 42s - loss: 7.6744 - accuracy: 0.4995
11840/25000 [=============>................] - ETA: 42s - loss: 7.6731 - accuracy: 0.4996
11872/25000 [=============>................] - ETA: 42s - loss: 7.6731 - accuracy: 0.4996
11904/25000 [=============>................] - ETA: 42s - loss: 7.6705 - accuracy: 0.4997
11936/25000 [=============>................] - ETA: 42s - loss: 7.6679 - accuracy: 0.4999
11968/25000 [=============>................] - ETA: 42s - loss: 7.6717 - accuracy: 0.4997
12000/25000 [=============>................] - ETA: 41s - loss: 7.6730 - accuracy: 0.4996
12032/25000 [=============>................] - ETA: 41s - loss: 7.6730 - accuracy: 0.4996
12064/25000 [=============>................] - ETA: 41s - loss: 7.6742 - accuracy: 0.4995
12096/25000 [=============>................] - ETA: 41s - loss: 7.6730 - accuracy: 0.4996
12128/25000 [=============>................] - ETA: 41s - loss: 7.6767 - accuracy: 0.4993
12160/25000 [=============>................] - ETA: 41s - loss: 7.6818 - accuracy: 0.4990
12192/25000 [=============>................] - ETA: 41s - loss: 7.6880 - accuracy: 0.4986
12224/25000 [=============>................] - ETA: 41s - loss: 7.6854 - accuracy: 0.4988
12256/25000 [=============>................] - ETA: 41s - loss: 7.6879 - accuracy: 0.4986
12288/25000 [=============>................] - ETA: 40s - loss: 7.6878 - accuracy: 0.4986
12320/25000 [=============>................] - ETA: 40s - loss: 7.6840 - accuracy: 0.4989
12352/25000 [=============>................] - ETA: 40s - loss: 7.6828 - accuracy: 0.4989
12384/25000 [=============>................] - ETA: 40s - loss: 7.6852 - accuracy: 0.4988
12416/25000 [=============>................] - ETA: 40s - loss: 7.6827 - accuracy: 0.4990
12448/25000 [=============>................] - ETA: 40s - loss: 7.6839 - accuracy: 0.4989
12480/25000 [=============>................] - ETA: 40s - loss: 7.6826 - accuracy: 0.4990
12512/25000 [==============>...............] - ETA: 40s - loss: 7.6875 - accuracy: 0.4986
12544/25000 [==============>...............] - ETA: 40s - loss: 7.6874 - accuracy: 0.4986
12576/25000 [==============>...............] - ETA: 40s - loss: 7.6886 - accuracy: 0.4986
12608/25000 [==============>...............] - ETA: 39s - loss: 7.6897 - accuracy: 0.4985
12640/25000 [==============>...............] - ETA: 39s - loss: 7.6848 - accuracy: 0.4988
12672/25000 [==============>...............] - ETA: 39s - loss: 7.6872 - accuracy: 0.4987
12704/25000 [==============>...............] - ETA: 39s - loss: 7.6896 - accuracy: 0.4985
12736/25000 [==============>...............] - ETA: 39s - loss: 7.6907 - accuracy: 0.4984
12768/25000 [==============>...............] - ETA: 39s - loss: 7.6894 - accuracy: 0.4985
12800/25000 [==============>...............] - ETA: 39s - loss: 7.6882 - accuracy: 0.4986
12832/25000 [==============>...............] - ETA: 39s - loss: 7.6845 - accuracy: 0.4988
12864/25000 [==============>...............] - ETA: 39s - loss: 7.6917 - accuracy: 0.4984
12896/25000 [==============>...............] - ETA: 39s - loss: 7.6928 - accuracy: 0.4983
12928/25000 [==============>...............] - ETA: 38s - loss: 7.6892 - accuracy: 0.4985
12960/25000 [==============>...............] - ETA: 38s - loss: 7.6796 - accuracy: 0.4992
12992/25000 [==============>...............] - ETA: 38s - loss: 7.6772 - accuracy: 0.4993
13024/25000 [==============>...............] - ETA: 38s - loss: 7.6784 - accuracy: 0.4992
13056/25000 [==============>...............] - ETA: 38s - loss: 7.6795 - accuracy: 0.4992
13088/25000 [==============>...............] - ETA: 38s - loss: 7.6807 - accuracy: 0.4991
13120/25000 [==============>...............] - ETA: 38s - loss: 7.6818 - accuracy: 0.4990
13152/25000 [==============>...............] - ETA: 38s - loss: 7.6748 - accuracy: 0.4995
13184/25000 [==============>...............] - ETA: 38s - loss: 7.6759 - accuracy: 0.4994
13216/25000 [==============>...............] - ETA: 38s - loss: 7.6794 - accuracy: 0.4992
13248/25000 [==============>...............] - ETA: 37s - loss: 7.6759 - accuracy: 0.4994
13280/25000 [==============>...............] - ETA: 37s - loss: 7.6735 - accuracy: 0.4995
13312/25000 [==============>...............] - ETA: 37s - loss: 7.6747 - accuracy: 0.4995
13344/25000 [===============>..............] - ETA: 37s - loss: 7.6735 - accuracy: 0.4996
13376/25000 [===============>..............] - ETA: 37s - loss: 7.6735 - accuracy: 0.4996
13408/25000 [===============>..............] - ETA: 37s - loss: 7.6735 - accuracy: 0.4996
13440/25000 [===============>..............] - ETA: 37s - loss: 7.6712 - accuracy: 0.4997
13472/25000 [===============>..............] - ETA: 37s - loss: 7.6723 - accuracy: 0.4996
13504/25000 [===============>..............] - ETA: 37s - loss: 7.6723 - accuracy: 0.4996
13536/25000 [===============>..............] - ETA: 36s - loss: 7.6791 - accuracy: 0.4992
13568/25000 [===============>..............] - ETA: 36s - loss: 7.6768 - accuracy: 0.4993
13600/25000 [===============>..............] - ETA: 36s - loss: 7.6790 - accuracy: 0.4992
13632/25000 [===============>..............] - ETA: 36s - loss: 7.6756 - accuracy: 0.4994
13664/25000 [===============>..............] - ETA: 36s - loss: 7.6767 - accuracy: 0.4993
13696/25000 [===============>..............] - ETA: 36s - loss: 7.6789 - accuracy: 0.4992
13728/25000 [===============>..............] - ETA: 36s - loss: 7.6767 - accuracy: 0.4993
13760/25000 [===============>..............] - ETA: 36s - loss: 7.6722 - accuracy: 0.4996
13792/25000 [===============>..............] - ETA: 36s - loss: 7.6722 - accuracy: 0.4996
13824/25000 [===============>..............] - ETA: 36s - loss: 7.6755 - accuracy: 0.4994
13856/25000 [===============>..............] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
13888/25000 [===============>..............] - ETA: 35s - loss: 7.6677 - accuracy: 0.4999
13920/25000 [===============>..............] - ETA: 35s - loss: 7.6743 - accuracy: 0.4995
13952/25000 [===============>..............] - ETA: 35s - loss: 7.6754 - accuracy: 0.4994
13984/25000 [===============>..............] - ETA: 35s - loss: 7.6776 - accuracy: 0.4993
14016/25000 [===============>..............] - ETA: 35s - loss: 7.6743 - accuracy: 0.4995
14048/25000 [===============>..............] - ETA: 35s - loss: 7.6775 - accuracy: 0.4993
14080/25000 [===============>..............] - ETA: 35s - loss: 7.6742 - accuracy: 0.4995
14112/25000 [===============>..............] - ETA: 35s - loss: 7.6764 - accuracy: 0.4994
14144/25000 [===============>..............] - ETA: 34s - loss: 7.6775 - accuracy: 0.4993
14176/25000 [================>.............] - ETA: 34s - loss: 7.6731 - accuracy: 0.4996
14208/25000 [================>.............] - ETA: 34s - loss: 7.6731 - accuracy: 0.4996
14240/25000 [================>.............] - ETA: 34s - loss: 7.6709 - accuracy: 0.4997
14272/25000 [================>.............] - ETA: 34s - loss: 7.6720 - accuracy: 0.4996
14304/25000 [================>.............] - ETA: 34s - loss: 7.6677 - accuracy: 0.4999
14336/25000 [================>.............] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
14368/25000 [================>.............] - ETA: 34s - loss: 7.6645 - accuracy: 0.5001
14400/25000 [================>.............] - ETA: 34s - loss: 7.6677 - accuracy: 0.4999
14432/25000 [================>.............] - ETA: 34s - loss: 7.6687 - accuracy: 0.4999
14464/25000 [================>.............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
14496/25000 [================>.............] - ETA: 33s - loss: 7.6645 - accuracy: 0.5001
14528/25000 [================>.............] - ETA: 33s - loss: 7.6635 - accuracy: 0.5002
14560/25000 [================>.............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
14592/25000 [================>.............] - ETA: 33s - loss: 7.6656 - accuracy: 0.5001
14624/25000 [================>.............] - ETA: 33s - loss: 7.6645 - accuracy: 0.5001
14656/25000 [================>.............] - ETA: 33s - loss: 7.6635 - accuracy: 0.5002
14688/25000 [================>.............] - ETA: 33s - loss: 7.6645 - accuracy: 0.5001
14720/25000 [================>.............] - ETA: 33s - loss: 7.6687 - accuracy: 0.4999
14752/25000 [================>.............] - ETA: 32s - loss: 7.6697 - accuracy: 0.4998
14784/25000 [================>.............] - ETA: 32s - loss: 7.6708 - accuracy: 0.4997
14816/25000 [================>.............] - ETA: 32s - loss: 7.6718 - accuracy: 0.4997
14848/25000 [================>.............] - ETA: 32s - loss: 7.6769 - accuracy: 0.4993
14880/25000 [================>.............] - ETA: 32s - loss: 7.6749 - accuracy: 0.4995
14912/25000 [================>.............] - ETA: 32s - loss: 7.6759 - accuracy: 0.4994
14944/25000 [================>.............] - ETA: 32s - loss: 7.6779 - accuracy: 0.4993
14976/25000 [================>.............] - ETA: 32s - loss: 7.6758 - accuracy: 0.4994
15008/25000 [=================>............] - ETA: 32s - loss: 7.6748 - accuracy: 0.4995
15040/25000 [=================>............] - ETA: 32s - loss: 7.6768 - accuracy: 0.4993
15072/25000 [=================>............] - ETA: 31s - loss: 7.6737 - accuracy: 0.4995
15104/25000 [=================>............] - ETA: 31s - loss: 7.6697 - accuracy: 0.4998
15136/25000 [=================>............] - ETA: 31s - loss: 7.6727 - accuracy: 0.4996
15168/25000 [=================>............] - ETA: 31s - loss: 7.6727 - accuracy: 0.4996
15200/25000 [=================>............] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
15232/25000 [=================>............] - ETA: 31s - loss: 7.6686 - accuracy: 0.4999
15264/25000 [=================>............] - ETA: 31s - loss: 7.6716 - accuracy: 0.4997
15296/25000 [=================>............] - ETA: 31s - loss: 7.6716 - accuracy: 0.4997
15328/25000 [=================>............] - ETA: 31s - loss: 7.6726 - accuracy: 0.4996
15360/25000 [=================>............] - ETA: 31s - loss: 7.6716 - accuracy: 0.4997
15392/25000 [=================>............] - ETA: 30s - loss: 7.6696 - accuracy: 0.4998
15424/25000 [=================>............] - ETA: 30s - loss: 7.6686 - accuracy: 0.4999
15456/25000 [=================>............] - ETA: 30s - loss: 7.6706 - accuracy: 0.4997
15488/25000 [=================>............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
15520/25000 [=================>............] - ETA: 30s - loss: 7.6656 - accuracy: 0.5001
15552/25000 [=================>............] - ETA: 30s - loss: 7.6696 - accuracy: 0.4998
15584/25000 [=================>............] - ETA: 30s - loss: 7.6755 - accuracy: 0.4994
15616/25000 [=================>............] - ETA: 30s - loss: 7.6794 - accuracy: 0.4992
15648/25000 [=================>............] - ETA: 30s - loss: 7.6803 - accuracy: 0.4991
15680/25000 [=================>............] - ETA: 29s - loss: 7.6813 - accuracy: 0.4990
15712/25000 [=================>............] - ETA: 29s - loss: 7.6783 - accuracy: 0.4992
15744/25000 [=================>............] - ETA: 29s - loss: 7.6754 - accuracy: 0.4994
15776/25000 [=================>............] - ETA: 29s - loss: 7.6763 - accuracy: 0.4994
15808/25000 [=================>............] - ETA: 29s - loss: 7.6802 - accuracy: 0.4991
15840/25000 [==================>...........] - ETA: 29s - loss: 7.6860 - accuracy: 0.4987
15872/25000 [==================>...........] - ETA: 29s - loss: 7.6869 - accuracy: 0.4987
15904/25000 [==================>...........] - ETA: 29s - loss: 7.6869 - accuracy: 0.4987
15936/25000 [==================>...........] - ETA: 29s - loss: 7.6849 - accuracy: 0.4988
15968/25000 [==================>...........] - ETA: 29s - loss: 7.6829 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 28s - loss: 7.6820 - accuracy: 0.4990
16032/25000 [==================>...........] - ETA: 28s - loss: 7.6867 - accuracy: 0.4987
16064/25000 [==================>...........] - ETA: 28s - loss: 7.6876 - accuracy: 0.4986
16096/25000 [==================>...........] - ETA: 28s - loss: 7.6847 - accuracy: 0.4988
16128/25000 [==================>...........] - ETA: 28s - loss: 7.6866 - accuracy: 0.4987
16160/25000 [==================>...........] - ETA: 28s - loss: 7.6865 - accuracy: 0.4987
16192/25000 [==================>...........] - ETA: 28s - loss: 7.6875 - accuracy: 0.4986
16224/25000 [==================>...........] - ETA: 28s - loss: 7.6884 - accuracy: 0.4986
16256/25000 [==================>...........] - ETA: 28s - loss: 7.6874 - accuracy: 0.4986
16288/25000 [==================>...........] - ETA: 28s - loss: 7.6873 - accuracy: 0.4986
16320/25000 [==================>...........] - ETA: 27s - loss: 7.6826 - accuracy: 0.4990
16352/25000 [==================>...........] - ETA: 27s - loss: 7.6807 - accuracy: 0.4991
16384/25000 [==================>...........] - ETA: 27s - loss: 7.6807 - accuracy: 0.4991
16416/25000 [==================>...........] - ETA: 27s - loss: 7.6769 - accuracy: 0.4993
16448/25000 [==================>...........] - ETA: 27s - loss: 7.6778 - accuracy: 0.4993
16480/25000 [==================>...........] - ETA: 27s - loss: 7.6778 - accuracy: 0.4993
16512/25000 [==================>...........] - ETA: 27s - loss: 7.6833 - accuracy: 0.4989
16544/25000 [==================>...........] - ETA: 27s - loss: 7.6852 - accuracy: 0.4988
16576/25000 [==================>...........] - ETA: 27s - loss: 7.6805 - accuracy: 0.4991
16608/25000 [==================>...........] - ETA: 27s - loss: 7.6805 - accuracy: 0.4991
16640/25000 [==================>...........] - ETA: 26s - loss: 7.6832 - accuracy: 0.4989
16672/25000 [===================>..........] - ETA: 26s - loss: 7.6878 - accuracy: 0.4986
16704/25000 [===================>..........] - ETA: 26s - loss: 7.6877 - accuracy: 0.4986
16736/25000 [===================>..........] - ETA: 26s - loss: 7.6877 - accuracy: 0.4986
16768/25000 [===================>..........] - ETA: 26s - loss: 7.6877 - accuracy: 0.4986
16800/25000 [===================>..........] - ETA: 26s - loss: 7.6903 - accuracy: 0.4985
16832/25000 [===================>..........] - ETA: 26s - loss: 7.6930 - accuracy: 0.4983
16864/25000 [===================>..........] - ETA: 26s - loss: 7.6957 - accuracy: 0.4981
16896/25000 [===================>..........] - ETA: 26s - loss: 7.6938 - accuracy: 0.4982
16928/25000 [===================>..........] - ETA: 26s - loss: 7.6920 - accuracy: 0.4983
16960/25000 [===================>..........] - ETA: 25s - loss: 7.6946 - accuracy: 0.4982
16992/25000 [===================>..........] - ETA: 25s - loss: 7.6937 - accuracy: 0.4982
17024/25000 [===================>..........] - ETA: 25s - loss: 7.6945 - accuracy: 0.4982
17056/25000 [===================>..........] - ETA: 25s - loss: 7.6927 - accuracy: 0.4983
17088/25000 [===================>..........] - ETA: 25s - loss: 7.6944 - accuracy: 0.4982
17120/25000 [===================>..........] - ETA: 25s - loss: 7.6935 - accuracy: 0.4982
17152/25000 [===================>..........] - ETA: 25s - loss: 7.6943 - accuracy: 0.4982
17184/25000 [===================>..........] - ETA: 25s - loss: 7.6934 - accuracy: 0.4983
17216/25000 [===================>..........] - ETA: 25s - loss: 7.6951 - accuracy: 0.4981
17248/25000 [===================>..........] - ETA: 24s - loss: 7.6968 - accuracy: 0.4980
17280/25000 [===================>..........] - ETA: 24s - loss: 7.6950 - accuracy: 0.4981
17312/25000 [===================>..........] - ETA: 24s - loss: 7.6932 - accuracy: 0.4983
17344/25000 [===================>..........] - ETA: 24s - loss: 7.6949 - accuracy: 0.4982
17376/25000 [===================>..........] - ETA: 24s - loss: 7.6931 - accuracy: 0.4983
17408/25000 [===================>..........] - ETA: 24s - loss: 7.6922 - accuracy: 0.4983
17440/25000 [===================>..........] - ETA: 24s - loss: 7.6921 - accuracy: 0.4983
17472/25000 [===================>..........] - ETA: 24s - loss: 7.6947 - accuracy: 0.4982
17504/25000 [====================>.........] - ETA: 24s - loss: 7.6920 - accuracy: 0.4983
17536/25000 [====================>.........] - ETA: 24s - loss: 7.6929 - accuracy: 0.4983
17568/25000 [====================>.........] - ETA: 23s - loss: 7.6928 - accuracy: 0.4983
17600/25000 [====================>.........] - ETA: 23s - loss: 7.6910 - accuracy: 0.4984
17632/25000 [====================>.........] - ETA: 23s - loss: 7.6901 - accuracy: 0.4985
17664/25000 [====================>.........] - ETA: 23s - loss: 7.6927 - accuracy: 0.4983
17696/25000 [====================>.........] - ETA: 23s - loss: 7.6909 - accuracy: 0.4984
17728/25000 [====================>.........] - ETA: 23s - loss: 7.6908 - accuracy: 0.4984
17760/25000 [====================>.........] - ETA: 23s - loss: 7.6882 - accuracy: 0.4986
17792/25000 [====================>.........] - ETA: 23s - loss: 7.6864 - accuracy: 0.4987
17824/25000 [====================>.........] - ETA: 23s - loss: 7.6795 - accuracy: 0.4992
17856/25000 [====================>.........] - ETA: 23s - loss: 7.6829 - accuracy: 0.4989
17888/25000 [====================>.........] - ETA: 22s - loss: 7.6820 - accuracy: 0.4990
17920/25000 [====================>.........] - ETA: 22s - loss: 7.6829 - accuracy: 0.4989
17952/25000 [====================>.........] - ETA: 22s - loss: 7.6871 - accuracy: 0.4987
17984/25000 [====================>.........] - ETA: 22s - loss: 7.6879 - accuracy: 0.4986
18016/25000 [====================>.........] - ETA: 22s - loss: 7.6887 - accuracy: 0.4986
18048/25000 [====================>.........] - ETA: 22s - loss: 7.6862 - accuracy: 0.4987
18080/25000 [====================>.........] - ETA: 22s - loss: 7.6878 - accuracy: 0.4986
18112/25000 [====================>.........] - ETA: 22s - loss: 7.6878 - accuracy: 0.4986
18144/25000 [====================>.........] - ETA: 22s - loss: 7.6903 - accuracy: 0.4985
18176/25000 [====================>.........] - ETA: 21s - loss: 7.6860 - accuracy: 0.4987
18208/25000 [====================>.........] - ETA: 21s - loss: 7.6843 - accuracy: 0.4988
18240/25000 [====================>.........] - ETA: 21s - loss: 7.6860 - accuracy: 0.4987
18272/25000 [====================>.........] - ETA: 21s - loss: 7.6826 - accuracy: 0.4990
18304/25000 [====================>.........] - ETA: 21s - loss: 7.6825 - accuracy: 0.4990
18336/25000 [=====================>........] - ETA: 21s - loss: 7.6808 - accuracy: 0.4991
18368/25000 [=====================>........] - ETA: 21s - loss: 7.6800 - accuracy: 0.4991
18400/25000 [=====================>........] - ETA: 21s - loss: 7.6808 - accuracy: 0.4991
18432/25000 [=====================>........] - ETA: 21s - loss: 7.6766 - accuracy: 0.4993
18464/25000 [=====================>........] - ETA: 21s - loss: 7.6782 - accuracy: 0.4992
18496/25000 [=====================>........] - ETA: 20s - loss: 7.6824 - accuracy: 0.4990
18528/25000 [=====================>........] - ETA: 20s - loss: 7.6790 - accuracy: 0.4992
18560/25000 [=====================>........] - ETA: 20s - loss: 7.6807 - accuracy: 0.4991
18592/25000 [=====================>........] - ETA: 20s - loss: 7.6790 - accuracy: 0.4992
18624/25000 [=====================>........] - ETA: 20s - loss: 7.6831 - accuracy: 0.4989
18656/25000 [=====================>........] - ETA: 20s - loss: 7.6839 - accuracy: 0.4989
18688/25000 [=====================>........] - ETA: 20s - loss: 7.6822 - accuracy: 0.4990
18720/25000 [=====================>........] - ETA: 20s - loss: 7.6838 - accuracy: 0.4989
18752/25000 [=====================>........] - ETA: 20s - loss: 7.6813 - accuracy: 0.4990
18784/25000 [=====================>........] - ETA: 20s - loss: 7.6797 - accuracy: 0.4991
18816/25000 [=====================>........] - ETA: 19s - loss: 7.6862 - accuracy: 0.4987
18848/25000 [=====================>........] - ETA: 19s - loss: 7.6870 - accuracy: 0.4987
18880/25000 [=====================>........] - ETA: 19s - loss: 7.6885 - accuracy: 0.4986
18912/25000 [=====================>........] - ETA: 19s - loss: 7.6893 - accuracy: 0.4985
18944/25000 [=====================>........] - ETA: 19s - loss: 7.6893 - accuracy: 0.4985
18976/25000 [=====================>........] - ETA: 19s - loss: 7.6917 - accuracy: 0.4984
19008/25000 [=====================>........] - ETA: 19s - loss: 7.6932 - accuracy: 0.4983
19040/25000 [=====================>........] - ETA: 19s - loss: 7.6948 - accuracy: 0.4982
19072/25000 [=====================>........] - ETA: 19s - loss: 7.6940 - accuracy: 0.4982
19104/25000 [=====================>........] - ETA: 18s - loss: 7.6955 - accuracy: 0.4981
19136/25000 [=====================>........] - ETA: 18s - loss: 7.6923 - accuracy: 0.4983
19168/25000 [======================>.......] - ETA: 18s - loss: 7.6954 - accuracy: 0.4981
19200/25000 [======================>.......] - ETA: 18s - loss: 7.6954 - accuracy: 0.4981
19232/25000 [======================>.......] - ETA: 18s - loss: 7.6969 - accuracy: 0.4980
19264/25000 [======================>.......] - ETA: 18s - loss: 7.6985 - accuracy: 0.4979
19296/25000 [======================>.......] - ETA: 18s - loss: 7.6984 - accuracy: 0.4979
19328/25000 [======================>.......] - ETA: 18s - loss: 7.6976 - accuracy: 0.4980
19360/25000 [======================>.......] - ETA: 18s - loss: 7.6983 - accuracy: 0.4979
19392/25000 [======================>.......] - ETA: 18s - loss: 7.6998 - accuracy: 0.4978
19424/25000 [======================>.......] - ETA: 17s - loss: 7.6990 - accuracy: 0.4979
19456/25000 [======================>.......] - ETA: 17s - loss: 7.7013 - accuracy: 0.4977
19488/25000 [======================>.......] - ETA: 17s - loss: 7.7020 - accuracy: 0.4977
19520/25000 [======================>.......] - ETA: 17s - loss: 7.7012 - accuracy: 0.4977
19552/25000 [======================>.......] - ETA: 17s - loss: 7.7003 - accuracy: 0.4978
19584/25000 [======================>.......] - ETA: 17s - loss: 7.7003 - accuracy: 0.4978
19616/25000 [======================>.......] - ETA: 17s - loss: 7.7026 - accuracy: 0.4977
19648/25000 [======================>.......] - ETA: 17s - loss: 7.7017 - accuracy: 0.4977
19680/25000 [======================>.......] - ETA: 17s - loss: 7.7025 - accuracy: 0.4977
19712/25000 [======================>.......] - ETA: 17s - loss: 7.7016 - accuracy: 0.4977
19744/25000 [======================>.......] - ETA: 16s - loss: 7.6992 - accuracy: 0.4979
19776/25000 [======================>.......] - ETA: 16s - loss: 7.6976 - accuracy: 0.4980
19808/25000 [======================>.......] - ETA: 16s - loss: 7.6984 - accuracy: 0.4979
19840/25000 [======================>.......] - ETA: 16s - loss: 7.7022 - accuracy: 0.4977
19872/25000 [======================>.......] - ETA: 16s - loss: 7.7029 - accuracy: 0.4976
19904/25000 [======================>.......] - ETA: 16s - loss: 7.7036 - accuracy: 0.4976
19936/25000 [======================>.......] - ETA: 16s - loss: 7.7051 - accuracy: 0.4975
19968/25000 [======================>.......] - ETA: 16s - loss: 7.7019 - accuracy: 0.4977
20000/25000 [=======================>......] - ETA: 16s - loss: 7.7027 - accuracy: 0.4976
20032/25000 [=======================>......] - ETA: 15s - loss: 7.7011 - accuracy: 0.4978
20064/25000 [=======================>......] - ETA: 15s - loss: 7.7018 - accuracy: 0.4977
20096/25000 [=======================>......] - ETA: 15s - loss: 7.7032 - accuracy: 0.4976
20128/25000 [=======================>......] - ETA: 15s - loss: 7.7062 - accuracy: 0.4974
20160/25000 [=======================>......] - ETA: 15s - loss: 7.7039 - accuracy: 0.4976
20192/25000 [=======================>......] - ETA: 15s - loss: 7.7023 - accuracy: 0.4977
20224/25000 [=======================>......] - ETA: 15s - loss: 7.7007 - accuracy: 0.4978
20256/25000 [=======================>......] - ETA: 15s - loss: 7.6992 - accuracy: 0.4979
20288/25000 [=======================>......] - ETA: 15s - loss: 7.7006 - accuracy: 0.4978
20320/25000 [=======================>......] - ETA: 15s - loss: 7.7021 - accuracy: 0.4977
20352/25000 [=======================>......] - ETA: 14s - loss: 7.7005 - accuracy: 0.4978
20384/25000 [=======================>......] - ETA: 14s - loss: 7.6975 - accuracy: 0.4980
20416/25000 [=======================>......] - ETA: 14s - loss: 7.6997 - accuracy: 0.4978
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6966 - accuracy: 0.4980
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6988 - accuracy: 0.4979
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6973 - accuracy: 0.4980
20544/25000 [=======================>......] - ETA: 14s - loss: 7.7017 - accuracy: 0.4977
20576/25000 [=======================>......] - ETA: 14s - loss: 7.7039 - accuracy: 0.4976
20608/25000 [=======================>......] - ETA: 14s - loss: 7.7031 - accuracy: 0.4976
20640/25000 [=======================>......] - ETA: 14s - loss: 7.7008 - accuracy: 0.4978
20672/25000 [=======================>......] - ETA: 13s - loss: 7.6978 - accuracy: 0.4980
20704/25000 [=======================>......] - ETA: 13s - loss: 7.6992 - accuracy: 0.4979
20736/25000 [=======================>......] - ETA: 13s - loss: 7.7036 - accuracy: 0.4976
20768/25000 [=======================>......] - ETA: 13s - loss: 7.7072 - accuracy: 0.4974
20800/25000 [=======================>......] - ETA: 13s - loss: 7.7057 - accuracy: 0.4975
20832/25000 [=======================>......] - ETA: 13s - loss: 7.7056 - accuracy: 0.4975
20864/25000 [========================>.....] - ETA: 13s - loss: 7.7070 - accuracy: 0.4974
20896/25000 [========================>.....] - ETA: 13s - loss: 7.7040 - accuracy: 0.4976
20928/25000 [========================>.....] - ETA: 13s - loss: 7.7076 - accuracy: 0.4973
20960/25000 [========================>.....] - ETA: 12s - loss: 7.7090 - accuracy: 0.4972
20992/25000 [========================>.....] - ETA: 12s - loss: 7.7112 - accuracy: 0.4971
21024/25000 [========================>.....] - ETA: 12s - loss: 7.7104 - accuracy: 0.4971
21056/25000 [========================>.....] - ETA: 12s - loss: 7.7081 - accuracy: 0.4973
21088/25000 [========================>.....] - ETA: 12s - loss: 7.7095 - accuracy: 0.4972
21120/25000 [========================>.....] - ETA: 12s - loss: 7.7058 - accuracy: 0.4974
21152/25000 [========================>.....] - ETA: 12s - loss: 7.7079 - accuracy: 0.4973
21184/25000 [========================>.....] - ETA: 12s - loss: 7.7079 - accuracy: 0.4973
21216/25000 [========================>.....] - ETA: 12s - loss: 7.7093 - accuracy: 0.4972
21248/25000 [========================>.....] - ETA: 12s - loss: 7.7099 - accuracy: 0.4972
21280/25000 [========================>.....] - ETA: 11s - loss: 7.7120 - accuracy: 0.4970
21312/25000 [========================>.....] - ETA: 11s - loss: 7.7091 - accuracy: 0.4972
21344/25000 [========================>.....] - ETA: 11s - loss: 7.7076 - accuracy: 0.4973
21376/25000 [========================>.....] - ETA: 11s - loss: 7.7068 - accuracy: 0.4974
21408/25000 [========================>.....] - ETA: 11s - loss: 7.7082 - accuracy: 0.4973
21440/25000 [========================>.....] - ETA: 11s - loss: 7.7102 - accuracy: 0.4972
21472/25000 [========================>.....] - ETA: 11s - loss: 7.7080 - accuracy: 0.4973
21504/25000 [========================>.....] - ETA: 11s - loss: 7.7087 - accuracy: 0.4973
21536/25000 [========================>.....] - ETA: 11s - loss: 7.7079 - accuracy: 0.4973
21568/25000 [========================>.....] - ETA: 11s - loss: 7.7064 - accuracy: 0.4974
21600/25000 [========================>.....] - ETA: 10s - loss: 7.7092 - accuracy: 0.4972
21632/25000 [========================>.....] - ETA: 10s - loss: 7.7077 - accuracy: 0.4973
21664/25000 [========================>.....] - ETA: 10s - loss: 7.7077 - accuracy: 0.4973
21696/25000 [=========================>....] - ETA: 10s - loss: 7.7083 - accuracy: 0.4973
21728/25000 [=========================>....] - ETA: 10s - loss: 7.7083 - accuracy: 0.4973
21760/25000 [=========================>....] - ETA: 10s - loss: 7.7054 - accuracy: 0.4975
21792/25000 [=========================>....] - ETA: 10s - loss: 7.7074 - accuracy: 0.4973
21824/25000 [=========================>....] - ETA: 10s - loss: 7.7060 - accuracy: 0.4974
21856/25000 [=========================>....] - ETA: 10s - loss: 7.7038 - accuracy: 0.4976
21888/25000 [=========================>....] - ETA: 10s - loss: 7.7023 - accuracy: 0.4977
21920/25000 [=========================>....] - ETA: 9s - loss: 7.7086 - accuracy: 0.4973 
21952/25000 [=========================>....] - ETA: 9s - loss: 7.7064 - accuracy: 0.4974
21984/25000 [=========================>....] - ETA: 9s - loss: 7.7085 - accuracy: 0.4973
22016/25000 [=========================>....] - ETA: 9s - loss: 7.7077 - accuracy: 0.4973
22048/25000 [=========================>....] - ETA: 9s - loss: 7.7083 - accuracy: 0.4973
22080/25000 [=========================>....] - ETA: 9s - loss: 7.7048 - accuracy: 0.4975
22112/25000 [=========================>....] - ETA: 9s - loss: 7.7027 - accuracy: 0.4976
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6999 - accuracy: 0.4978
22176/25000 [=========================>....] - ETA: 9s - loss: 7.7012 - accuracy: 0.4977
22208/25000 [=========================>....] - ETA: 8s - loss: 7.6991 - accuracy: 0.4979
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6963 - accuracy: 0.4981
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6976 - accuracy: 0.4980
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6976 - accuracy: 0.4980
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6975 - accuracy: 0.4980
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6954 - accuracy: 0.4981
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6961 - accuracy: 0.4981
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6933 - accuracy: 0.4983
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6953 - accuracy: 0.4981
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6939 - accuracy: 0.4982
22528/25000 [==========================>...] - ETA: 7s - loss: 7.6932 - accuracy: 0.4983
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6897 - accuracy: 0.4985
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6897 - accuracy: 0.4985
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6876 - accuracy: 0.4986
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6876 - accuracy: 0.4986
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6889 - accuracy: 0.4985
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6848 - accuracy: 0.4988
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6868 - accuracy: 0.4987
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6868 - accuracy: 0.4987
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6875 - accuracy: 0.4986
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6847 - accuracy: 0.4988
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6834 - accuracy: 0.4989
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6813 - accuracy: 0.4990
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6806 - accuracy: 0.4991
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6793 - accuracy: 0.4992
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6746 - accuracy: 0.4995
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6726 - accuracy: 0.4996
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6733 - accuracy: 0.4996
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6719 - accuracy: 0.4997
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6759 - accuracy: 0.4994
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6765 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6745 - accuracy: 0.4995
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6732 - accuracy: 0.4996
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6719 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6724 - accuracy: 0.4996
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24256/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24288/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24320/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24352/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24384/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24448/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24480/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24512/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24640/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24672/25000 [============================>.] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
24704/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
24768/25000 [============================>.] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
24800/25000 [============================>.] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
24832/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24864/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 96s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

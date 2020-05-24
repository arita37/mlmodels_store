
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:52<01:18, 26.19s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.20851734522587906, 'embedding_size_factor': 0.910507047431182, 'layers.choice': 0, 'learning_rate': 0.0007275082429461443, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.3739163368042068e-07} and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xca\xb0\xb2E2k\x10X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed"\xdf\xac\xef\xd2\x84X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?G\xd6\xc8\x0e6\t\xc2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x82p\xbd\x97\x1f\x9c\x97u.' and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xca\xb0\xb2E2k\x10X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed"\xdf\xac\xef\xd2\x84X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?G\xd6\xc8\x0e6\t\xc2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x82p\xbd\x97\x1f\x9c\x97u.' and reward: 0.3718
 60%|██████    | 3/5 [01:44<01:07, 33.86s/it] 60%|██████    | 3/5 [01:44<01:09, 34.71s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.11523182325993961, 'embedding_size_factor': 1.2173202630723607, 'layers.choice': 3, 'learning_rate': 0.00021269183105837578, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.9008258155176085e-08} and reward: 0.3538
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xbd\x7f\xd50\\!\xeeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3z$\xcf\xeavfX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?+\xe0\xc0\xea\xc3g\xfdX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>Th\xf3A`\xe2Iu.' and reward: 0.3538
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xbd\x7f\xd50\\!\xeeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3z$\xcf\xeavfX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?+\xe0\xc0\xea\xc3g\xfdX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>Th\xf3A`\xe2Iu.' and reward: 0.3538
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 161.54919052124023
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -44.21s of remaining time.
Ensemble size: 25
Ensemble weights: 
[0.72 0.2  0.08]
	0.3912	 = Validation accuracy score
	1.02s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 165.27s ...
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
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
   90112/17464789 [..............................] - ETA: 38s
  180224/17464789 [..............................] - ETA: 25s
  335872/17464789 [..............................] - ETA: 17s
  647168/17464789 [>.............................] - ETA: 10s
 1269760/17464789 [=>............................] - ETA: 5s 
 2514944/17464789 [===>..........................] - ETA: 3s
 4988928/17464789 [=======>......................] - ETA: 1s
 7938048/17464789 [============>.................] - ETA: 0s
11051008/17464789 [=================>............] - ETA: 0s
14049280/17464789 [=======================>......] - ETA: 0s
17063936/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-24 10:21:28.373172: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-24 10:21:28.377444: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-24 10:21:28.377642: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5608e5fec920 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 10:21:28.377659: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:14 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:50 - loss: 9.1041 - accuracy: 0.4062
   96/25000 [..............................] - ETA: 2:19 - loss: 9.1041 - accuracy: 0.4062
  128/25000 [..............................] - ETA: 2:04 - loss: 8.6249 - accuracy: 0.4375
  160/25000 [..............................] - ETA: 1:53 - loss: 8.4333 - accuracy: 0.4500
  192/25000 [..............................] - ETA: 1:46 - loss: 7.9861 - accuracy: 0.4792
  224/25000 [..............................] - ETA: 1:41 - loss: 7.8720 - accuracy: 0.4866
  256/25000 [..............................] - ETA: 1:38 - loss: 7.8463 - accuracy: 0.4883
  288/25000 [..............................] - ETA: 1:35 - loss: 8.0393 - accuracy: 0.4757
  320/25000 [..............................] - ETA: 1:33 - loss: 8.0979 - accuracy: 0.4719
  352/25000 [..............................] - ETA: 1:31 - loss: 8.2329 - accuracy: 0.4631
  384/25000 [..............................] - ETA: 1:30 - loss: 8.1458 - accuracy: 0.4688
  416/25000 [..............................] - ETA: 1:29 - loss: 8.2932 - accuracy: 0.4591
  448/25000 [..............................] - ETA: 1:28 - loss: 8.2485 - accuracy: 0.4621
  480/25000 [..............................] - ETA: 1:27 - loss: 8.2416 - accuracy: 0.4625
  512/25000 [..............................] - ETA: 1:26 - loss: 8.2656 - accuracy: 0.4609
  544/25000 [..............................] - ETA: 1:25 - loss: 8.1740 - accuracy: 0.4669
  576/25000 [..............................] - ETA: 1:24 - loss: 8.1192 - accuracy: 0.4705
  608/25000 [..............................] - ETA: 1:23 - loss: 8.0953 - accuracy: 0.4720
  640/25000 [..............................] - ETA: 1:23 - loss: 8.1218 - accuracy: 0.4703
  672/25000 [..............................] - ETA: 1:22 - loss: 8.0089 - accuracy: 0.4777
  704/25000 [..............................] - ETA: 1:22 - loss: 7.9498 - accuracy: 0.4815
  736/25000 [..............................] - ETA: 1:21 - loss: 7.9583 - accuracy: 0.4810
  768/25000 [..............................] - ETA: 1:21 - loss: 8.0060 - accuracy: 0.4779
  800/25000 [..............................] - ETA: 1:20 - loss: 7.9733 - accuracy: 0.4800
  832/25000 [..............................] - ETA: 1:20 - loss: 7.9431 - accuracy: 0.4820
  864/25000 [>.............................] - ETA: 1:19 - loss: 7.9328 - accuracy: 0.4826
  896/25000 [>.............................] - ETA: 1:19 - loss: 8.0602 - accuracy: 0.4743
  928/25000 [>.............................] - ETA: 1:19 - loss: 8.1293 - accuracy: 0.4698
  960/25000 [>.............................] - ETA: 1:18 - loss: 8.0979 - accuracy: 0.4719
  992/25000 [>.............................] - ETA: 1:18 - loss: 7.9603 - accuracy: 0.4808
 1024/25000 [>.............................] - ETA: 1:18 - loss: 7.9361 - accuracy: 0.4824
 1056/25000 [>.............................] - ETA: 1:17 - loss: 7.9425 - accuracy: 0.4820
 1088/25000 [>.............................] - ETA: 1:17 - loss: 7.8921 - accuracy: 0.4853
 1120/25000 [>.............................] - ETA: 1:17 - loss: 7.8994 - accuracy: 0.4848
 1152/25000 [>.............................] - ETA: 1:17 - loss: 7.9328 - accuracy: 0.4826
 1184/25000 [>.............................] - ETA: 1:16 - loss: 7.9127 - accuracy: 0.4840
 1216/25000 [>.............................] - ETA: 1:16 - loss: 7.8936 - accuracy: 0.4852
 1248/25000 [>.............................] - ETA: 1:16 - loss: 7.9123 - accuracy: 0.4840
 1280/25000 [>.............................] - ETA: 1:16 - loss: 7.8942 - accuracy: 0.4852
 1312/25000 [>.............................] - ETA: 1:15 - loss: 7.8653 - accuracy: 0.4870
 1344/25000 [>.............................] - ETA: 1:15 - loss: 7.8377 - accuracy: 0.4888
 1376/25000 [>.............................] - ETA: 1:15 - loss: 7.8672 - accuracy: 0.4869
 1408/25000 [>.............................] - ETA: 1:15 - loss: 7.8735 - accuracy: 0.4865
 1440/25000 [>.............................] - ETA: 1:15 - loss: 7.9009 - accuracy: 0.4847
 1472/25000 [>.............................] - ETA: 1:15 - loss: 7.9166 - accuracy: 0.4837
 1504/25000 [>.............................] - ETA: 1:14 - loss: 7.8909 - accuracy: 0.4854
 1536/25000 [>.............................] - ETA: 1:14 - loss: 7.9062 - accuracy: 0.4844
 1568/25000 [>.............................] - ETA: 1:14 - loss: 7.9209 - accuracy: 0.4834
 1600/25000 [>.............................] - ETA: 1:14 - loss: 7.9062 - accuracy: 0.4844
 1632/25000 [>.............................] - ETA: 1:13 - loss: 7.9579 - accuracy: 0.4810
 1664/25000 [>.............................] - ETA: 1:13 - loss: 7.9523 - accuracy: 0.4814
 1696/25000 [=>............................] - ETA: 1:13 - loss: 7.9107 - accuracy: 0.4841
 1728/25000 [=>............................] - ETA: 1:13 - loss: 7.8530 - accuracy: 0.4878
 1760/25000 [=>............................] - ETA: 1:13 - loss: 7.8583 - accuracy: 0.4875
 1792/25000 [=>............................] - ETA: 1:13 - loss: 7.8805 - accuracy: 0.4860
 1824/25000 [=>............................] - ETA: 1:13 - loss: 7.8347 - accuracy: 0.4890
 1856/25000 [=>............................] - ETA: 1:12 - loss: 7.8153 - accuracy: 0.4903
 1888/25000 [=>............................] - ETA: 1:12 - loss: 7.8372 - accuracy: 0.4889
 1920/25000 [=>............................] - ETA: 1:12 - loss: 7.8024 - accuracy: 0.4911
 1952/25000 [=>............................] - ETA: 1:12 - loss: 7.8002 - accuracy: 0.4913
 1984/25000 [=>............................] - ETA: 1:12 - loss: 7.7671 - accuracy: 0.4934
 2016/25000 [=>............................] - ETA: 1:12 - loss: 7.7655 - accuracy: 0.4936
 2048/25000 [=>............................] - ETA: 1:12 - loss: 7.8014 - accuracy: 0.4912
 2080/25000 [=>............................] - ETA: 1:11 - loss: 7.7993 - accuracy: 0.4913
 2112/25000 [=>............................] - ETA: 1:11 - loss: 7.8191 - accuracy: 0.4901
 2144/25000 [=>............................] - ETA: 1:11 - loss: 7.8240 - accuracy: 0.4897
 2176/25000 [=>............................] - ETA: 1:11 - loss: 7.8498 - accuracy: 0.4881
 2208/25000 [=>............................] - ETA: 1:11 - loss: 7.8611 - accuracy: 0.4873
 2240/25000 [=>............................] - ETA: 1:11 - loss: 7.8514 - accuracy: 0.4879
 2272/25000 [=>............................] - ETA: 1:10 - loss: 7.8488 - accuracy: 0.4881
 2304/25000 [=>............................] - ETA: 1:10 - loss: 7.8263 - accuracy: 0.4896
 2336/25000 [=>............................] - ETA: 1:10 - loss: 7.8110 - accuracy: 0.4906
 2368/25000 [=>............................] - ETA: 1:10 - loss: 7.8155 - accuracy: 0.4903
 2400/25000 [=>............................] - ETA: 1:10 - loss: 7.8072 - accuracy: 0.4908
 2432/25000 [=>............................] - ETA: 1:10 - loss: 7.8053 - accuracy: 0.4910
 2464/25000 [=>............................] - ETA: 1:10 - loss: 7.7849 - accuracy: 0.4923
 2496/25000 [=>............................] - ETA: 1:09 - loss: 7.7588 - accuracy: 0.4940
 2528/25000 [==>...........................] - ETA: 1:09 - loss: 7.7819 - accuracy: 0.4925
 2560/25000 [==>...........................] - ETA: 1:09 - loss: 7.7744 - accuracy: 0.4930
 2592/25000 [==>...........................] - ETA: 1:09 - loss: 7.7554 - accuracy: 0.4942
 2624/25000 [==>...........................] - ETA: 1:09 - loss: 7.7660 - accuracy: 0.4935
 2656/25000 [==>...........................] - ETA: 1:09 - loss: 7.7474 - accuracy: 0.4947
 2688/25000 [==>...........................] - ETA: 1:09 - loss: 7.7408 - accuracy: 0.4952
 2720/25000 [==>...........................] - ETA: 1:08 - loss: 7.7568 - accuracy: 0.4941
 2752/25000 [==>...........................] - ETA: 1:08 - loss: 7.7613 - accuracy: 0.4938
 2784/25000 [==>...........................] - ETA: 1:08 - loss: 7.7547 - accuracy: 0.4943
 2816/25000 [==>...........................] - ETA: 1:08 - loss: 7.7646 - accuracy: 0.4936
 2848/25000 [==>...........................] - ETA: 1:08 - loss: 7.7528 - accuracy: 0.4944
 2880/25000 [==>...........................] - ETA: 1:08 - loss: 7.7625 - accuracy: 0.4938
 2912/25000 [==>...........................] - ETA: 1:08 - loss: 7.7509 - accuracy: 0.4945
 2944/25000 [==>...........................] - ETA: 1:08 - loss: 7.7291 - accuracy: 0.4959
 2976/25000 [==>...........................] - ETA: 1:07 - loss: 7.7284 - accuracy: 0.4960
 3008/25000 [==>...........................] - ETA: 1:07 - loss: 7.7227 - accuracy: 0.4963
 3040/25000 [==>...........................] - ETA: 1:07 - loss: 7.7271 - accuracy: 0.4961
 3072/25000 [==>...........................] - ETA: 1:07 - loss: 7.7365 - accuracy: 0.4954
 3104/25000 [==>...........................] - ETA: 1:07 - loss: 7.7308 - accuracy: 0.4958
 3136/25000 [==>...........................] - ETA: 1:07 - loss: 7.7302 - accuracy: 0.4959
 3168/25000 [==>...........................] - ETA: 1:07 - loss: 7.7392 - accuracy: 0.4953
 3200/25000 [==>...........................] - ETA: 1:06 - loss: 7.7529 - accuracy: 0.4944
 3232/25000 [==>...........................] - ETA: 1:06 - loss: 7.7235 - accuracy: 0.4963
 3264/25000 [==>...........................] - ETA: 1:06 - loss: 7.7136 - accuracy: 0.4969
 3296/25000 [==>...........................] - ETA: 1:06 - loss: 7.7317 - accuracy: 0.4958
 3328/25000 [==>...........................] - ETA: 1:06 - loss: 7.7219 - accuracy: 0.4964
 3360/25000 [===>..........................] - ETA: 1:06 - loss: 7.7031 - accuracy: 0.4976
 3392/25000 [===>..........................] - ETA: 1:06 - loss: 7.6892 - accuracy: 0.4985
 3424/25000 [===>..........................] - ETA: 1:06 - loss: 7.7024 - accuracy: 0.4977
 3456/25000 [===>..........................] - ETA: 1:05 - loss: 7.7065 - accuracy: 0.4974
 3488/25000 [===>..........................] - ETA: 1:05 - loss: 7.6842 - accuracy: 0.4989
 3520/25000 [===>..........................] - ETA: 1:05 - loss: 7.6884 - accuracy: 0.4986
 3552/25000 [===>..........................] - ETA: 1:06 - loss: 7.6968 - accuracy: 0.4980
 3584/25000 [===>..........................] - ETA: 1:05 - loss: 7.6923 - accuracy: 0.4983
 3616/25000 [===>..........................] - ETA: 1:05 - loss: 7.6793 - accuracy: 0.4992
 3648/25000 [===>..........................] - ETA: 1:05 - loss: 7.6750 - accuracy: 0.4995
 3680/25000 [===>..........................] - ETA: 1:05 - loss: 7.6833 - accuracy: 0.4989
 3712/25000 [===>..........................] - ETA: 1:05 - loss: 7.6790 - accuracy: 0.4992
 3744/25000 [===>..........................] - ETA: 1:05 - loss: 7.6789 - accuracy: 0.4992
 3776/25000 [===>..........................] - ETA: 1:05 - loss: 7.7032 - accuracy: 0.4976
 3808/25000 [===>..........................] - ETA: 1:05 - loss: 7.7109 - accuracy: 0.4971
 3840/25000 [===>..........................] - ETA: 1:04 - loss: 7.7065 - accuracy: 0.4974
 3872/25000 [===>..........................] - ETA: 1:04 - loss: 7.6943 - accuracy: 0.4982
 3904/25000 [===>..........................] - ETA: 1:04 - loss: 7.6980 - accuracy: 0.4980
 3936/25000 [===>..........................] - ETA: 1:04 - loss: 7.7212 - accuracy: 0.4964
 3968/25000 [===>..........................] - ETA: 1:04 - loss: 7.7130 - accuracy: 0.4970
 4000/25000 [===>..........................] - ETA: 1:04 - loss: 7.7126 - accuracy: 0.4970
 4032/25000 [===>..........................] - ETA: 1:04 - loss: 7.7008 - accuracy: 0.4978
 4064/25000 [===>..........................] - ETA: 1:04 - loss: 7.7119 - accuracy: 0.4970
 4096/25000 [===>..........................] - ETA: 1:03 - loss: 7.7003 - accuracy: 0.4978
 4128/25000 [===>..........................] - ETA: 1:03 - loss: 7.6926 - accuracy: 0.4983
 4160/25000 [===>..........................] - ETA: 1:03 - loss: 7.6924 - accuracy: 0.4983
 4192/25000 [====>.........................] - ETA: 1:03 - loss: 7.7032 - accuracy: 0.4976
 4224/25000 [====>.........................] - ETA: 1:03 - loss: 7.7211 - accuracy: 0.4964
 4256/25000 [====>.........................] - ETA: 1:03 - loss: 7.7099 - accuracy: 0.4972
 4288/25000 [====>.........................] - ETA: 1:03 - loss: 7.7024 - accuracy: 0.4977
 4320/25000 [====>.........................] - ETA: 1:03 - loss: 7.7092 - accuracy: 0.4972
 4352/25000 [====>.........................] - ETA: 1:02 - loss: 7.7054 - accuracy: 0.4975
 4384/25000 [====>.........................] - ETA: 1:02 - loss: 7.7086 - accuracy: 0.4973
 4416/25000 [====>.........................] - ETA: 1:02 - loss: 7.7118 - accuracy: 0.4971
 4448/25000 [====>.........................] - ETA: 1:02 - loss: 7.7114 - accuracy: 0.4971
 4480/25000 [====>.........................] - ETA: 1:02 - loss: 7.7248 - accuracy: 0.4962
 4512/25000 [====>.........................] - ETA: 1:02 - loss: 7.7176 - accuracy: 0.4967
 4544/25000 [====>.........................] - ETA: 1:02 - loss: 7.7240 - accuracy: 0.4963
 4576/25000 [====>.........................] - ETA: 1:02 - loss: 7.7303 - accuracy: 0.4958
 4608/25000 [====>.........................] - ETA: 1:02 - loss: 7.7265 - accuracy: 0.4961
 4640/25000 [====>.........................] - ETA: 1:01 - loss: 7.7162 - accuracy: 0.4968
 4672/25000 [====>.........................] - ETA: 1:01 - loss: 7.7257 - accuracy: 0.4961
 4704/25000 [====>.........................] - ETA: 1:01 - loss: 7.7123 - accuracy: 0.4970
 4736/25000 [====>.........................] - ETA: 1:01 - loss: 7.7087 - accuracy: 0.4973
 4768/25000 [====>.........................] - ETA: 1:01 - loss: 7.7020 - accuracy: 0.4977
 4800/25000 [====>.........................] - ETA: 1:01 - loss: 7.7145 - accuracy: 0.4969
 4832/25000 [====>.........................] - ETA: 1:01 - loss: 7.7142 - accuracy: 0.4969
 4864/25000 [====>.........................] - ETA: 1:01 - loss: 7.7202 - accuracy: 0.4965
 4896/25000 [====>.........................] - ETA: 1:01 - loss: 7.7230 - accuracy: 0.4963
 4928/25000 [====>.........................] - ETA: 1:00 - loss: 7.7320 - accuracy: 0.4957
 4960/25000 [====>.........................] - ETA: 1:00 - loss: 7.7192 - accuracy: 0.4966
 4992/25000 [====>.........................] - ETA: 1:00 - loss: 7.7096 - accuracy: 0.4972
 5024/25000 [=====>........................] - ETA: 1:00 - loss: 7.7124 - accuracy: 0.4970
 5056/25000 [=====>........................] - ETA: 1:00 - loss: 7.7273 - accuracy: 0.4960
 5088/25000 [=====>........................] - ETA: 1:00 - loss: 7.7359 - accuracy: 0.4955
 5120/25000 [=====>........................] - ETA: 1:00 - loss: 7.7385 - accuracy: 0.4953
 5152/25000 [=====>........................] - ETA: 1:00 - loss: 7.7291 - accuracy: 0.4959
 5184/25000 [=====>........................] - ETA: 1:00 - loss: 7.7287 - accuracy: 0.4959
 5216/25000 [=====>........................] - ETA: 1:00 - loss: 7.7313 - accuracy: 0.4958
 5248/25000 [=====>........................] - ETA: 59s - loss: 7.7397 - accuracy: 0.4952 
 5280/25000 [=====>........................] - ETA: 59s - loss: 7.7305 - accuracy: 0.4958
 5312/25000 [=====>........................] - ETA: 59s - loss: 7.7186 - accuracy: 0.4966
 5344/25000 [=====>........................] - ETA: 59s - loss: 7.7297 - accuracy: 0.4959
 5376/25000 [=====>........................] - ETA: 59s - loss: 7.7294 - accuracy: 0.4959
 5408/25000 [=====>........................] - ETA: 59s - loss: 7.7318 - accuracy: 0.4957
 5440/25000 [=====>........................] - ETA: 59s - loss: 7.7286 - accuracy: 0.4960
 5472/25000 [=====>........................] - ETA: 59s - loss: 7.7199 - accuracy: 0.4965
 5504/25000 [=====>........................] - ETA: 59s - loss: 7.7196 - accuracy: 0.4965
 5536/25000 [=====>........................] - ETA: 58s - loss: 7.7276 - accuracy: 0.4960
 5568/25000 [=====>........................] - ETA: 58s - loss: 7.7107 - accuracy: 0.4971
 5600/25000 [=====>........................] - ETA: 58s - loss: 7.7104 - accuracy: 0.4971
 5632/25000 [=====>........................] - ETA: 58s - loss: 7.7129 - accuracy: 0.4970
 5664/25000 [=====>........................] - ETA: 58s - loss: 7.7099 - accuracy: 0.4972
 5696/25000 [=====>........................] - ETA: 58s - loss: 7.7097 - accuracy: 0.4972
 5728/25000 [=====>........................] - ETA: 58s - loss: 7.7121 - accuracy: 0.4970
 5760/25000 [=====>........................] - ETA: 58s - loss: 7.7092 - accuracy: 0.4972
 5792/25000 [=====>........................] - ETA: 58s - loss: 7.7169 - accuracy: 0.4967
 5824/25000 [=====>........................] - ETA: 58s - loss: 7.7193 - accuracy: 0.4966
 5856/25000 [======>.......................] - ETA: 57s - loss: 7.7111 - accuracy: 0.4971
 5888/25000 [======>.......................] - ETA: 57s - loss: 7.7213 - accuracy: 0.4964
 5920/25000 [======>.......................] - ETA: 57s - loss: 7.7210 - accuracy: 0.4965
 5952/25000 [======>.......................] - ETA: 57s - loss: 7.7336 - accuracy: 0.4956
 5984/25000 [======>.......................] - ETA: 57s - loss: 7.7204 - accuracy: 0.4965
 6016/25000 [======>.......................] - ETA: 57s - loss: 7.7125 - accuracy: 0.4970
 6048/25000 [======>.......................] - ETA: 57s - loss: 7.7249 - accuracy: 0.4962
 6080/25000 [======>.......................] - ETA: 57s - loss: 7.7297 - accuracy: 0.4959
 6112/25000 [======>.......................] - ETA: 57s - loss: 7.7193 - accuracy: 0.4966
 6144/25000 [======>.......................] - ETA: 56s - loss: 7.7140 - accuracy: 0.4969
 6176/25000 [======>.......................] - ETA: 56s - loss: 7.6939 - accuracy: 0.4982
 6208/25000 [======>.......................] - ETA: 56s - loss: 7.6987 - accuracy: 0.4979
 6240/25000 [======>.......................] - ETA: 56s - loss: 7.7133 - accuracy: 0.4970
 6272/25000 [======>.......................] - ETA: 56s - loss: 7.7155 - accuracy: 0.4968
 6304/25000 [======>.......................] - ETA: 56s - loss: 7.7177 - accuracy: 0.4967
 6336/25000 [======>.......................] - ETA: 56s - loss: 7.7320 - accuracy: 0.4957
 6368/25000 [======>.......................] - ETA: 56s - loss: 7.7268 - accuracy: 0.4961
 6400/25000 [======>.......................] - ETA: 56s - loss: 7.7169 - accuracy: 0.4967
 6432/25000 [======>.......................] - ETA: 56s - loss: 7.7191 - accuracy: 0.4966
 6464/25000 [======>.......................] - ETA: 55s - loss: 7.7164 - accuracy: 0.4968
 6496/25000 [======>.......................] - ETA: 55s - loss: 7.7138 - accuracy: 0.4969
 6528/25000 [======>.......................] - ETA: 55s - loss: 7.7136 - accuracy: 0.4969
 6560/25000 [======>.......................] - ETA: 55s - loss: 7.7087 - accuracy: 0.4973
 6592/25000 [======>.......................] - ETA: 55s - loss: 7.7038 - accuracy: 0.4976
 6624/25000 [======>.......................] - ETA: 55s - loss: 7.6990 - accuracy: 0.4979
 6656/25000 [======>.......................] - ETA: 55s - loss: 7.6989 - accuracy: 0.4979
 6688/25000 [=======>......................] - ETA: 55s - loss: 7.6873 - accuracy: 0.4987
 6720/25000 [=======>......................] - ETA: 55s - loss: 7.6963 - accuracy: 0.4981
 6752/25000 [=======>......................] - ETA: 54s - loss: 7.6984 - accuracy: 0.4979
 6784/25000 [=======>......................] - ETA: 54s - loss: 7.7096 - accuracy: 0.4972
 6816/25000 [=======>......................] - ETA: 54s - loss: 7.7116 - accuracy: 0.4971
 6848/25000 [=======>......................] - ETA: 54s - loss: 7.7047 - accuracy: 0.4975
 6880/25000 [=======>......................] - ETA: 54s - loss: 7.6934 - accuracy: 0.4983
 6912/25000 [=======>......................] - ETA: 54s - loss: 7.7043 - accuracy: 0.4975
 6944/25000 [=======>......................] - ETA: 54s - loss: 7.7108 - accuracy: 0.4971
 6976/25000 [=======>......................] - ETA: 54s - loss: 7.7194 - accuracy: 0.4966
 7008/25000 [=======>......................] - ETA: 54s - loss: 7.7235 - accuracy: 0.4963
 7040/25000 [=======>......................] - ETA: 54s - loss: 7.7211 - accuracy: 0.4964
 7072/25000 [=======>......................] - ETA: 53s - loss: 7.7230 - accuracy: 0.4963
 7104/25000 [=======>......................] - ETA: 53s - loss: 7.7206 - accuracy: 0.4965
 7136/25000 [=======>......................] - ETA: 53s - loss: 7.7160 - accuracy: 0.4968
 7168/25000 [=======>......................] - ETA: 53s - loss: 7.7180 - accuracy: 0.4967
 7200/25000 [=======>......................] - ETA: 53s - loss: 7.7028 - accuracy: 0.4976
 7232/25000 [=======>......................] - ETA: 53s - loss: 7.7069 - accuracy: 0.4974
 7264/25000 [=======>......................] - ETA: 53s - loss: 7.7004 - accuracy: 0.4978
 7296/25000 [=======>......................] - ETA: 53s - loss: 7.7023 - accuracy: 0.4977
 7328/25000 [=======>......................] - ETA: 53s - loss: 7.7022 - accuracy: 0.4977
 7360/25000 [=======>......................] - ETA: 53s - loss: 7.7104 - accuracy: 0.4971
 7392/25000 [=======>......................] - ETA: 52s - loss: 7.7081 - accuracy: 0.4973
 7424/25000 [=======>......................] - ETA: 52s - loss: 7.7059 - accuracy: 0.4974
 7456/25000 [=======>......................] - ETA: 52s - loss: 7.7098 - accuracy: 0.4972
 7488/25000 [=======>......................] - ETA: 52s - loss: 7.7178 - accuracy: 0.4967
 7520/25000 [========>.....................] - ETA: 52s - loss: 7.7176 - accuracy: 0.4967
 7552/25000 [========>.....................] - ETA: 52s - loss: 7.7133 - accuracy: 0.4970
 7584/25000 [========>.....................] - ETA: 52s - loss: 7.7151 - accuracy: 0.4968
 7616/25000 [========>.....................] - ETA: 52s - loss: 7.7149 - accuracy: 0.4968
 7648/25000 [========>.....................] - ETA: 52s - loss: 7.7107 - accuracy: 0.4971
 7680/25000 [========>.....................] - ETA: 52s - loss: 7.7125 - accuracy: 0.4970
 7712/25000 [========>.....................] - ETA: 51s - loss: 7.7243 - accuracy: 0.4962
 7744/25000 [========>.....................] - ETA: 51s - loss: 7.7280 - accuracy: 0.4960
 7776/25000 [========>.....................] - ETA: 51s - loss: 7.7258 - accuracy: 0.4961
 7808/25000 [========>.....................] - ETA: 51s - loss: 7.7196 - accuracy: 0.4965
 7840/25000 [========>.....................] - ETA: 51s - loss: 7.7136 - accuracy: 0.4969
 7872/25000 [========>.....................] - ETA: 51s - loss: 7.7153 - accuracy: 0.4968
 7904/25000 [========>.....................] - ETA: 51s - loss: 7.7132 - accuracy: 0.4970
 7936/25000 [========>.....................] - ETA: 51s - loss: 7.7169 - accuracy: 0.4967
 7968/25000 [========>.....................] - ETA: 51s - loss: 7.7263 - accuracy: 0.4961
 8000/25000 [========>.....................] - ETA: 51s - loss: 7.7165 - accuracy: 0.4967
 8032/25000 [========>.....................] - ETA: 50s - loss: 7.7124 - accuracy: 0.4970
 8064/25000 [========>.....................] - ETA: 50s - loss: 7.7218 - accuracy: 0.4964
 8096/25000 [========>.....................] - ETA: 50s - loss: 7.7121 - accuracy: 0.4970
 8128/25000 [========>.....................] - ETA: 50s - loss: 7.7081 - accuracy: 0.4973
 8160/25000 [========>.....................] - ETA: 50s - loss: 7.7080 - accuracy: 0.4973
 8192/25000 [========>.....................] - ETA: 50s - loss: 7.6966 - accuracy: 0.4980
 8224/25000 [========>.....................] - ETA: 50s - loss: 7.6909 - accuracy: 0.4984
 8256/25000 [========>.....................] - ETA: 50s - loss: 7.6870 - accuracy: 0.4987
 8288/25000 [========>.....................] - ETA: 50s - loss: 7.6851 - accuracy: 0.4988
 8320/25000 [========>.....................] - ETA: 50s - loss: 7.6758 - accuracy: 0.4994
 8352/25000 [=========>....................] - ETA: 49s - loss: 7.6758 - accuracy: 0.4994
 8384/25000 [=========>....................] - ETA: 49s - loss: 7.6776 - accuracy: 0.4993
 8416/25000 [=========>....................] - ETA: 49s - loss: 7.6757 - accuracy: 0.4994
 8448/25000 [=========>....................] - ETA: 49s - loss: 7.6757 - accuracy: 0.4994
 8480/25000 [=========>....................] - ETA: 49s - loss: 7.6847 - accuracy: 0.4988
 8512/25000 [=========>....................] - ETA: 49s - loss: 7.6828 - accuracy: 0.4989
 8544/25000 [=========>....................] - ETA: 49s - loss: 7.6828 - accuracy: 0.4989
 8576/25000 [=========>....................] - ETA: 49s - loss: 7.6809 - accuracy: 0.4991
 8608/25000 [=========>....................] - ETA: 49s - loss: 7.6773 - accuracy: 0.4993
 8640/25000 [=========>....................] - ETA: 49s - loss: 7.6790 - accuracy: 0.4992
 8672/25000 [=========>....................] - ETA: 48s - loss: 7.6737 - accuracy: 0.4995
 8704/25000 [=========>....................] - ETA: 48s - loss: 7.6684 - accuracy: 0.4999
 8736/25000 [=========>....................] - ETA: 48s - loss: 7.6631 - accuracy: 0.5002
 8768/25000 [=========>....................] - ETA: 48s - loss: 7.6649 - accuracy: 0.5001
 8800/25000 [=========>....................] - ETA: 48s - loss: 7.6649 - accuracy: 0.5001
 8832/25000 [=========>....................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 8864/25000 [=========>....................] - ETA: 48s - loss: 7.6683 - accuracy: 0.4999
 8896/25000 [=========>....................] - ETA: 48s - loss: 7.6614 - accuracy: 0.5003
 8928/25000 [=========>....................] - ETA: 48s - loss: 7.6580 - accuracy: 0.5006
 8960/25000 [=========>....................] - ETA: 48s - loss: 7.6529 - accuracy: 0.5009
 8992/25000 [=========>....................] - ETA: 47s - loss: 7.6479 - accuracy: 0.5012
 9024/25000 [=========>....................] - ETA: 47s - loss: 7.6564 - accuracy: 0.5007
 9056/25000 [=========>....................] - ETA: 47s - loss: 7.6514 - accuracy: 0.5010
 9088/25000 [=========>....................] - ETA: 47s - loss: 7.6531 - accuracy: 0.5009
 9120/25000 [=========>....................] - ETA: 47s - loss: 7.6565 - accuracy: 0.5007
 9152/25000 [=========>....................] - ETA: 47s - loss: 7.6549 - accuracy: 0.5008
 9184/25000 [==========>...................] - ETA: 47s - loss: 7.6533 - accuracy: 0.5009
 9216/25000 [==========>...................] - ETA: 47s - loss: 7.6583 - accuracy: 0.5005
 9248/25000 [==========>...................] - ETA: 47s - loss: 7.6550 - accuracy: 0.5008
 9280/25000 [==========>...................] - ETA: 47s - loss: 7.6567 - accuracy: 0.5006
 9312/25000 [==========>...................] - ETA: 46s - loss: 7.6600 - accuracy: 0.5004
 9344/25000 [==========>...................] - ETA: 46s - loss: 7.6650 - accuracy: 0.5001
 9376/25000 [==========>...................] - ETA: 46s - loss: 7.6633 - accuracy: 0.5002
 9408/25000 [==========>...................] - ETA: 46s - loss: 7.6666 - accuracy: 0.5000
 9440/25000 [==========>...................] - ETA: 46s - loss: 7.6666 - accuracy: 0.5000
 9472/25000 [==========>...................] - ETA: 46s - loss: 7.6618 - accuracy: 0.5003
 9504/25000 [==========>...................] - ETA: 46s - loss: 7.6698 - accuracy: 0.4998
 9536/25000 [==========>...................] - ETA: 46s - loss: 7.6763 - accuracy: 0.4994
 9568/25000 [==========>...................] - ETA: 46s - loss: 7.6810 - accuracy: 0.4991
 9600/25000 [==========>...................] - ETA: 46s - loss: 7.6826 - accuracy: 0.4990
 9632/25000 [==========>...................] - ETA: 46s - loss: 7.6841 - accuracy: 0.4989
 9664/25000 [==========>...................] - ETA: 45s - loss: 7.6841 - accuracy: 0.4989
 9696/25000 [==========>...................] - ETA: 45s - loss: 7.6856 - accuracy: 0.4988
 9728/25000 [==========>...................] - ETA: 45s - loss: 7.6777 - accuracy: 0.4993
 9760/25000 [==========>...................] - ETA: 45s - loss: 7.6792 - accuracy: 0.4992
 9792/25000 [==========>...................] - ETA: 45s - loss: 7.6823 - accuracy: 0.4990
 9824/25000 [==========>...................] - ETA: 45s - loss: 7.6838 - accuracy: 0.4989
 9856/25000 [==========>...................] - ETA: 45s - loss: 7.6900 - accuracy: 0.4985
 9888/25000 [==========>...................] - ETA: 45s - loss: 7.6899 - accuracy: 0.4985
 9920/25000 [==========>...................] - ETA: 45s - loss: 7.6883 - accuracy: 0.4986
 9952/25000 [==========>...................] - ETA: 45s - loss: 7.6851 - accuracy: 0.4988
 9984/25000 [==========>...................] - ETA: 44s - loss: 7.6835 - accuracy: 0.4989
10016/25000 [===========>..................] - ETA: 44s - loss: 7.6773 - accuracy: 0.4993
10048/25000 [===========>..................] - ETA: 44s - loss: 7.6742 - accuracy: 0.4995
10080/25000 [===========>..................] - ETA: 44s - loss: 7.6773 - accuracy: 0.4993
10112/25000 [===========>..................] - ETA: 44s - loss: 7.6788 - accuracy: 0.4992
10144/25000 [===========>..................] - ETA: 44s - loss: 7.6787 - accuracy: 0.4992
10176/25000 [===========>..................] - ETA: 44s - loss: 7.6847 - accuracy: 0.4988
10208/25000 [===========>..................] - ETA: 44s - loss: 7.6801 - accuracy: 0.4991
10240/25000 [===========>..................] - ETA: 44s - loss: 7.6786 - accuracy: 0.4992
10272/25000 [===========>..................] - ETA: 44s - loss: 7.6771 - accuracy: 0.4993
10304/25000 [===========>..................] - ETA: 43s - loss: 7.6755 - accuracy: 0.4994
10336/25000 [===========>..................] - ETA: 43s - loss: 7.6740 - accuracy: 0.4995
10368/25000 [===========>..................] - ETA: 43s - loss: 7.6711 - accuracy: 0.4997
10400/25000 [===========>..................] - ETA: 43s - loss: 7.6755 - accuracy: 0.4994
10432/25000 [===========>..................] - ETA: 43s - loss: 7.6813 - accuracy: 0.4990
10464/25000 [===========>..................] - ETA: 43s - loss: 7.6871 - accuracy: 0.4987
10496/25000 [===========>..................] - ETA: 43s - loss: 7.6827 - accuracy: 0.4990
10528/25000 [===========>..................] - ETA: 43s - loss: 7.6812 - accuracy: 0.4991
10560/25000 [===========>..................] - ETA: 43s - loss: 7.6753 - accuracy: 0.4994
10592/25000 [===========>..................] - ETA: 43s - loss: 7.6739 - accuracy: 0.4995
10624/25000 [===========>..................] - ETA: 42s - loss: 7.6738 - accuracy: 0.4995
10656/25000 [===========>..................] - ETA: 42s - loss: 7.6695 - accuracy: 0.4998
10688/25000 [===========>..................] - ETA: 42s - loss: 7.6709 - accuracy: 0.4997
10720/25000 [===========>..................] - ETA: 42s - loss: 7.6680 - accuracy: 0.4999
10752/25000 [===========>..................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
10784/25000 [===========>..................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
10816/25000 [===========>..................] - ETA: 42s - loss: 7.6652 - accuracy: 0.5001
10848/25000 [============>.................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
10880/25000 [============>.................] - ETA: 42s - loss: 7.6694 - accuracy: 0.4998
10912/25000 [============>.................] - ETA: 42s - loss: 7.6680 - accuracy: 0.4999
10944/25000 [============>.................] - ETA: 41s - loss: 7.6736 - accuracy: 0.4995
10976/25000 [============>.................] - ETA: 41s - loss: 7.6750 - accuracy: 0.4995
11008/25000 [============>.................] - ETA: 41s - loss: 7.6778 - accuracy: 0.4993
11040/25000 [============>.................] - ETA: 41s - loss: 7.6791 - accuracy: 0.4992
11072/25000 [============>.................] - ETA: 41s - loss: 7.6735 - accuracy: 0.4995
11104/25000 [============>.................] - ETA: 41s - loss: 7.6680 - accuracy: 0.4999
11136/25000 [============>.................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
11168/25000 [============>.................] - ETA: 41s - loss: 7.6584 - accuracy: 0.5005
11200/25000 [============>.................] - ETA: 41s - loss: 7.6598 - accuracy: 0.5004
11232/25000 [============>.................] - ETA: 41s - loss: 7.6598 - accuracy: 0.5004
11264/25000 [============>.................] - ETA: 41s - loss: 7.6639 - accuracy: 0.5002
11296/25000 [============>.................] - ETA: 40s - loss: 7.6653 - accuracy: 0.5001
11328/25000 [============>.................] - ETA: 40s - loss: 7.6639 - accuracy: 0.5002
11360/25000 [============>.................] - ETA: 40s - loss: 7.6639 - accuracy: 0.5002
11392/25000 [============>.................] - ETA: 40s - loss: 7.6612 - accuracy: 0.5004
11424/25000 [============>.................] - ETA: 40s - loss: 7.6653 - accuracy: 0.5001
11456/25000 [============>.................] - ETA: 40s - loss: 7.6733 - accuracy: 0.4996
11488/25000 [============>.................] - ETA: 40s - loss: 7.6760 - accuracy: 0.4994
11520/25000 [============>.................] - ETA: 40s - loss: 7.6733 - accuracy: 0.4996
11552/25000 [============>.................] - ETA: 40s - loss: 7.6693 - accuracy: 0.4998
11584/25000 [============>.................] - ETA: 40s - loss: 7.6706 - accuracy: 0.4997
11616/25000 [============>.................] - ETA: 39s - loss: 7.6719 - accuracy: 0.4997
11648/25000 [============>.................] - ETA: 39s - loss: 7.6719 - accuracy: 0.4997
11680/25000 [=============>................] - ETA: 39s - loss: 7.6850 - accuracy: 0.4988
11712/25000 [=============>................] - ETA: 39s - loss: 7.6836 - accuracy: 0.4989
11744/25000 [=============>................] - ETA: 39s - loss: 7.6914 - accuracy: 0.4984
11776/25000 [=============>................] - ETA: 39s - loss: 7.6940 - accuracy: 0.4982
11808/25000 [=============>................] - ETA: 39s - loss: 7.6978 - accuracy: 0.4980
11840/25000 [=============>................] - ETA: 39s - loss: 7.6951 - accuracy: 0.4981
11872/25000 [=============>................] - ETA: 39s - loss: 7.6963 - accuracy: 0.4981
11904/25000 [=============>................] - ETA: 39s - loss: 7.6911 - accuracy: 0.4984
11936/25000 [=============>................] - ETA: 38s - loss: 7.6859 - accuracy: 0.4987
11968/25000 [=============>................] - ETA: 38s - loss: 7.6884 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 38s - loss: 7.6935 - accuracy: 0.4983
12032/25000 [=============>................] - ETA: 38s - loss: 7.6896 - accuracy: 0.4985
12064/25000 [=============>................] - ETA: 38s - loss: 7.6908 - accuracy: 0.4984
12096/25000 [=============>................] - ETA: 38s - loss: 7.6932 - accuracy: 0.4983
12128/25000 [=============>................] - ETA: 38s - loss: 7.6970 - accuracy: 0.4980
12160/25000 [=============>................] - ETA: 38s - loss: 7.6944 - accuracy: 0.4982
12192/25000 [=============>................] - ETA: 38s - loss: 7.6943 - accuracy: 0.4982
12224/25000 [=============>................] - ETA: 38s - loss: 7.7030 - accuracy: 0.4976
12256/25000 [=============>................] - ETA: 38s - loss: 7.7079 - accuracy: 0.4973
12288/25000 [=============>................] - ETA: 37s - loss: 7.7128 - accuracy: 0.4970
12320/25000 [=============>................] - ETA: 37s - loss: 7.7176 - accuracy: 0.4967
12352/25000 [=============>................] - ETA: 37s - loss: 7.7150 - accuracy: 0.4968
12384/25000 [=============>................] - ETA: 37s - loss: 7.7186 - accuracy: 0.4966
12416/25000 [=============>................] - ETA: 37s - loss: 7.7234 - accuracy: 0.4963
12448/25000 [=============>................] - ETA: 37s - loss: 7.7257 - accuracy: 0.4961
12480/25000 [=============>................] - ETA: 37s - loss: 7.7281 - accuracy: 0.4960
12512/25000 [==============>...............] - ETA: 37s - loss: 7.7267 - accuracy: 0.4961
12544/25000 [==============>...............] - ETA: 37s - loss: 7.7253 - accuracy: 0.4962
12576/25000 [==============>...............] - ETA: 37s - loss: 7.7276 - accuracy: 0.4960
12608/25000 [==============>...............] - ETA: 36s - loss: 7.7311 - accuracy: 0.4958
12640/25000 [==============>...............] - ETA: 36s - loss: 7.7321 - accuracy: 0.4957
12672/25000 [==============>...............] - ETA: 36s - loss: 7.7259 - accuracy: 0.4961
12704/25000 [==============>...............] - ETA: 36s - loss: 7.7185 - accuracy: 0.4966
12736/25000 [==============>...............] - ETA: 36s - loss: 7.7208 - accuracy: 0.4965
12768/25000 [==============>...............] - ETA: 36s - loss: 7.7195 - accuracy: 0.4966
12800/25000 [==============>...............] - ETA: 36s - loss: 7.7217 - accuracy: 0.4964
12832/25000 [==============>...............] - ETA: 36s - loss: 7.7204 - accuracy: 0.4965
12864/25000 [==============>...............] - ETA: 36s - loss: 7.7262 - accuracy: 0.4961
12896/25000 [==============>...............] - ETA: 36s - loss: 7.7261 - accuracy: 0.4961
12928/25000 [==============>...............] - ETA: 36s - loss: 7.7295 - accuracy: 0.4959
12960/25000 [==============>...............] - ETA: 35s - loss: 7.7293 - accuracy: 0.4959
12992/25000 [==============>...............] - ETA: 35s - loss: 7.7292 - accuracy: 0.4959
13024/25000 [==============>...............] - ETA: 35s - loss: 7.7243 - accuracy: 0.4962
13056/25000 [==============>...............] - ETA: 35s - loss: 7.7242 - accuracy: 0.4962
13088/25000 [==============>...............] - ETA: 35s - loss: 7.7205 - accuracy: 0.4965
13120/25000 [==============>...............] - ETA: 35s - loss: 7.7274 - accuracy: 0.4960
13152/25000 [==============>...............] - ETA: 35s - loss: 7.7307 - accuracy: 0.4958
13184/25000 [==============>...............] - ETA: 35s - loss: 7.7341 - accuracy: 0.4956
13216/25000 [==============>...............] - ETA: 35s - loss: 7.7362 - accuracy: 0.4955
13248/25000 [==============>...............] - ETA: 35s - loss: 7.7349 - accuracy: 0.4955
13280/25000 [==============>...............] - ETA: 34s - loss: 7.7347 - accuracy: 0.4956
13312/25000 [==============>...............] - ETA: 34s - loss: 7.7334 - accuracy: 0.4956
13344/25000 [===============>..............] - ETA: 34s - loss: 7.7333 - accuracy: 0.4957
13376/25000 [===============>..............] - ETA: 34s - loss: 7.7365 - accuracy: 0.4954
13408/25000 [===============>..............] - ETA: 34s - loss: 7.7398 - accuracy: 0.4952
13440/25000 [===============>..............] - ETA: 34s - loss: 7.7385 - accuracy: 0.4953
13472/25000 [===============>..............] - ETA: 34s - loss: 7.7406 - accuracy: 0.4952
13504/25000 [===============>..............] - ETA: 34s - loss: 7.7404 - accuracy: 0.4952
13536/25000 [===============>..............] - ETA: 34s - loss: 7.7357 - accuracy: 0.4955
13568/25000 [===============>..............] - ETA: 34s - loss: 7.7344 - accuracy: 0.4956
13600/25000 [===============>..............] - ETA: 33s - loss: 7.7354 - accuracy: 0.4955
13632/25000 [===============>..............] - ETA: 33s - loss: 7.7341 - accuracy: 0.4956
13664/25000 [===============>..............] - ETA: 33s - loss: 7.7362 - accuracy: 0.4955
13696/25000 [===============>..............] - ETA: 33s - loss: 7.7338 - accuracy: 0.4956
13728/25000 [===============>..............] - ETA: 33s - loss: 7.7258 - accuracy: 0.4961
13760/25000 [===============>..............] - ETA: 33s - loss: 7.7246 - accuracy: 0.4962
13792/25000 [===============>..............] - ETA: 33s - loss: 7.7244 - accuracy: 0.4962
13824/25000 [===============>..............] - ETA: 33s - loss: 7.7232 - accuracy: 0.4963
13856/25000 [===============>..............] - ETA: 33s - loss: 7.7242 - accuracy: 0.4962
13888/25000 [===============>..............] - ETA: 33s - loss: 7.7196 - accuracy: 0.4965
13920/25000 [===============>..............] - ETA: 33s - loss: 7.7184 - accuracy: 0.4966
13952/25000 [===============>..............] - ETA: 32s - loss: 7.7128 - accuracy: 0.4970
13984/25000 [===============>..............] - ETA: 32s - loss: 7.7171 - accuracy: 0.4967
14016/25000 [===============>..............] - ETA: 32s - loss: 7.7169 - accuracy: 0.4967
14048/25000 [===============>..............] - ETA: 32s - loss: 7.7212 - accuracy: 0.4964
14080/25000 [===============>..............] - ETA: 32s - loss: 7.7200 - accuracy: 0.4965
14112/25000 [===============>..............] - ETA: 32s - loss: 7.7166 - accuracy: 0.4967
14144/25000 [===============>..............] - ETA: 32s - loss: 7.7208 - accuracy: 0.4965
14176/25000 [================>.............] - ETA: 32s - loss: 7.7229 - accuracy: 0.4963
14208/25000 [================>.............] - ETA: 32s - loss: 7.7238 - accuracy: 0.4963
14240/25000 [================>.............] - ETA: 32s - loss: 7.7226 - accuracy: 0.4963
14272/25000 [================>.............] - ETA: 31s - loss: 7.7257 - accuracy: 0.4961
14304/25000 [================>.............] - ETA: 31s - loss: 7.7245 - accuracy: 0.4962
14336/25000 [================>.............] - ETA: 31s - loss: 7.7265 - accuracy: 0.4961
14368/25000 [================>.............] - ETA: 31s - loss: 7.7242 - accuracy: 0.4962
14400/25000 [================>.............] - ETA: 31s - loss: 7.7209 - accuracy: 0.4965
14432/25000 [================>.............] - ETA: 31s - loss: 7.7240 - accuracy: 0.4963
14464/25000 [================>.............] - ETA: 31s - loss: 7.7186 - accuracy: 0.4966
14496/25000 [================>.............] - ETA: 31s - loss: 7.7206 - accuracy: 0.4965
14528/25000 [================>.............] - ETA: 31s - loss: 7.7215 - accuracy: 0.4964
14560/25000 [================>.............] - ETA: 31s - loss: 7.7214 - accuracy: 0.4964
14592/25000 [================>.............] - ETA: 31s - loss: 7.7202 - accuracy: 0.4965
14624/25000 [================>.............] - ETA: 30s - loss: 7.7190 - accuracy: 0.4966
14656/25000 [================>.............] - ETA: 30s - loss: 7.7189 - accuracy: 0.4966
14688/25000 [================>.............] - ETA: 30s - loss: 7.7188 - accuracy: 0.4966
14720/25000 [================>.............] - ETA: 30s - loss: 7.7177 - accuracy: 0.4967
14752/25000 [================>.............] - ETA: 30s - loss: 7.7155 - accuracy: 0.4968
14784/25000 [================>.............] - ETA: 30s - loss: 7.7164 - accuracy: 0.4968
14816/25000 [================>.............] - ETA: 30s - loss: 7.7173 - accuracy: 0.4967
14848/25000 [================>.............] - ETA: 30s - loss: 7.7131 - accuracy: 0.4970
14880/25000 [================>.............] - ETA: 30s - loss: 7.7109 - accuracy: 0.4971
14912/25000 [================>.............] - ETA: 30s - loss: 7.7077 - accuracy: 0.4973
14944/25000 [================>.............] - ETA: 29s - loss: 7.7118 - accuracy: 0.4971
14976/25000 [================>.............] - ETA: 29s - loss: 7.7158 - accuracy: 0.4968
15008/25000 [=================>............] - ETA: 29s - loss: 7.7116 - accuracy: 0.4971
15040/25000 [=================>............] - ETA: 29s - loss: 7.7074 - accuracy: 0.4973
15072/25000 [=================>............] - ETA: 29s - loss: 7.7083 - accuracy: 0.4973
15104/25000 [=================>............] - ETA: 29s - loss: 7.7082 - accuracy: 0.4973
15136/25000 [=================>............] - ETA: 29s - loss: 7.7082 - accuracy: 0.4973
15168/25000 [=================>............] - ETA: 29s - loss: 7.7081 - accuracy: 0.4973
15200/25000 [=================>............] - ETA: 29s - loss: 7.7080 - accuracy: 0.4973
15232/25000 [=================>............] - ETA: 29s - loss: 7.7159 - accuracy: 0.4968
15264/25000 [=================>............] - ETA: 28s - loss: 7.7168 - accuracy: 0.4967
15296/25000 [=================>............] - ETA: 28s - loss: 7.7117 - accuracy: 0.4971
15328/25000 [=================>............] - ETA: 28s - loss: 7.7066 - accuracy: 0.4974
15360/25000 [=================>............] - ETA: 28s - loss: 7.7085 - accuracy: 0.4973
15392/25000 [=================>............] - ETA: 28s - loss: 7.7124 - accuracy: 0.4970
15424/25000 [=================>............] - ETA: 28s - loss: 7.7114 - accuracy: 0.4971
15456/25000 [=================>............] - ETA: 28s - loss: 7.7083 - accuracy: 0.4973
15488/25000 [=================>............] - ETA: 28s - loss: 7.7092 - accuracy: 0.4972
15520/25000 [=================>............] - ETA: 28s - loss: 7.7081 - accuracy: 0.4973
15552/25000 [=================>............] - ETA: 28s - loss: 7.7031 - accuracy: 0.4976
15584/25000 [=================>............] - ETA: 28s - loss: 7.7011 - accuracy: 0.4978
15616/25000 [=================>............] - ETA: 27s - loss: 7.7020 - accuracy: 0.4977
15648/25000 [=================>............] - ETA: 27s - loss: 7.7039 - accuracy: 0.4976
15680/25000 [=================>............] - ETA: 27s - loss: 7.7018 - accuracy: 0.4977
15712/25000 [=================>............] - ETA: 27s - loss: 7.7018 - accuracy: 0.4977
15744/25000 [=================>............] - ETA: 27s - loss: 7.7017 - accuracy: 0.4977
15776/25000 [=================>............] - ETA: 27s - loss: 7.7055 - accuracy: 0.4975
15808/25000 [=================>............] - ETA: 27s - loss: 7.7054 - accuracy: 0.4975
15840/25000 [==================>...........] - ETA: 27s - loss: 7.6995 - accuracy: 0.4979
15872/25000 [==================>...........] - ETA: 27s - loss: 7.6995 - accuracy: 0.4979
15904/25000 [==================>...........] - ETA: 27s - loss: 7.6984 - accuracy: 0.4979
15936/25000 [==================>...........] - ETA: 26s - loss: 7.7013 - accuracy: 0.4977
15968/25000 [==================>...........] - ETA: 26s - loss: 7.7050 - accuracy: 0.4975
16000/25000 [==================>...........] - ETA: 26s - loss: 7.7050 - accuracy: 0.4975
16032/25000 [==================>...........] - ETA: 26s - loss: 7.7049 - accuracy: 0.4975
16064/25000 [==================>...........] - ETA: 26s - loss: 7.7067 - accuracy: 0.4974
16096/25000 [==================>...........] - ETA: 26s - loss: 7.7066 - accuracy: 0.4974
16128/25000 [==================>...........] - ETA: 26s - loss: 7.7113 - accuracy: 0.4971
16160/25000 [==================>...........] - ETA: 26s - loss: 7.7093 - accuracy: 0.4972
16192/25000 [==================>...........] - ETA: 26s - loss: 7.7111 - accuracy: 0.4971
16224/25000 [==================>...........] - ETA: 26s - loss: 7.7082 - accuracy: 0.4973
16256/25000 [==================>...........] - ETA: 26s - loss: 7.7072 - accuracy: 0.4974
16288/25000 [==================>...........] - ETA: 25s - loss: 7.7052 - accuracy: 0.4975
16320/25000 [==================>...........] - ETA: 25s - loss: 7.7080 - accuracy: 0.4973
16352/25000 [==================>...........] - ETA: 25s - loss: 7.7107 - accuracy: 0.4971
16384/25000 [==================>...........] - ETA: 25s - loss: 7.7097 - accuracy: 0.4972
16416/25000 [==================>...........] - ETA: 25s - loss: 7.7049 - accuracy: 0.4975
16448/25000 [==================>...........] - ETA: 25s - loss: 7.7011 - accuracy: 0.4978
16480/25000 [==================>...........] - ETA: 25s - loss: 7.6973 - accuracy: 0.4980
16512/25000 [==================>...........] - ETA: 25s - loss: 7.6954 - accuracy: 0.4981
16544/25000 [==================>...........] - ETA: 25s - loss: 7.6963 - accuracy: 0.4981
16576/25000 [==================>...........] - ETA: 25s - loss: 7.6971 - accuracy: 0.4980
16608/25000 [==================>...........] - ETA: 24s - loss: 7.6989 - accuracy: 0.4979
16640/25000 [==================>...........] - ETA: 24s - loss: 7.6998 - accuracy: 0.4978
16672/25000 [===================>..........] - ETA: 24s - loss: 7.6951 - accuracy: 0.4981
16704/25000 [===================>..........] - ETA: 24s - loss: 7.6960 - accuracy: 0.4981
16736/25000 [===================>..........] - ETA: 24s - loss: 7.6941 - accuracy: 0.4982
16768/25000 [===================>..........] - ETA: 24s - loss: 7.6931 - accuracy: 0.4983
16800/25000 [===================>..........] - ETA: 24s - loss: 7.6931 - accuracy: 0.4983
16832/25000 [===================>..........] - ETA: 24s - loss: 7.6903 - accuracy: 0.4985
16864/25000 [===================>..........] - ETA: 24s - loss: 7.6930 - accuracy: 0.4983
16896/25000 [===================>..........] - ETA: 24s - loss: 7.6948 - accuracy: 0.4982
16928/25000 [===================>..........] - ETA: 24s - loss: 7.6956 - accuracy: 0.4981
16960/25000 [===================>..........] - ETA: 23s - loss: 7.6937 - accuracy: 0.4982
16992/25000 [===================>..........] - ETA: 23s - loss: 7.6928 - accuracy: 0.4983
17024/25000 [===================>..........] - ETA: 23s - loss: 7.6882 - accuracy: 0.4986
17056/25000 [===================>..........] - ETA: 23s - loss: 7.6900 - accuracy: 0.4985
17088/25000 [===================>..........] - ETA: 23s - loss: 7.6899 - accuracy: 0.4985
17120/25000 [===================>..........] - ETA: 23s - loss: 7.6899 - accuracy: 0.4985
17152/25000 [===================>..........] - ETA: 23s - loss: 7.6872 - accuracy: 0.4987
17184/25000 [===================>..........] - ETA: 23s - loss: 7.6863 - accuracy: 0.4987
17216/25000 [===================>..........] - ETA: 23s - loss: 7.6844 - accuracy: 0.4988
17248/25000 [===================>..........] - ETA: 23s - loss: 7.6862 - accuracy: 0.4987
17280/25000 [===================>..........] - ETA: 22s - loss: 7.6844 - accuracy: 0.4988
17312/25000 [===================>..........] - ETA: 22s - loss: 7.6870 - accuracy: 0.4987
17344/25000 [===================>..........] - ETA: 22s - loss: 7.6878 - accuracy: 0.4986
17376/25000 [===================>..........] - ETA: 22s - loss: 7.6860 - accuracy: 0.4987
17408/25000 [===================>..........] - ETA: 22s - loss: 7.6860 - accuracy: 0.4987
17440/25000 [===================>..........] - ETA: 22s - loss: 7.6842 - accuracy: 0.4989
17472/25000 [===================>..........] - ETA: 22s - loss: 7.6824 - accuracy: 0.4990
17504/25000 [====================>.........] - ETA: 22s - loss: 7.6850 - accuracy: 0.4988
17536/25000 [====================>.........] - ETA: 22s - loss: 7.6824 - accuracy: 0.4990
17568/25000 [====================>.........] - ETA: 22s - loss: 7.6788 - accuracy: 0.4992
17600/25000 [====================>.........] - ETA: 22s - loss: 7.6797 - accuracy: 0.4991
17632/25000 [====================>.........] - ETA: 21s - loss: 7.6805 - accuracy: 0.4991
17664/25000 [====================>.........] - ETA: 21s - loss: 7.6762 - accuracy: 0.4994
17696/25000 [====================>.........] - ETA: 21s - loss: 7.6762 - accuracy: 0.4994
17728/25000 [====================>.........] - ETA: 21s - loss: 7.6735 - accuracy: 0.4995
17760/25000 [====================>.........] - ETA: 21s - loss: 7.6718 - accuracy: 0.4997
17792/25000 [====================>.........] - ETA: 21s - loss: 7.6727 - accuracy: 0.4996
17824/25000 [====================>.........] - ETA: 21s - loss: 7.6709 - accuracy: 0.4997
17856/25000 [====================>.........] - ETA: 21s - loss: 7.6675 - accuracy: 0.4999
17888/25000 [====================>.........] - ETA: 21s - loss: 7.6658 - accuracy: 0.5001
17920/25000 [====================>.........] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
17952/25000 [====================>.........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
17984/25000 [====================>.........] - ETA: 20s - loss: 7.6649 - accuracy: 0.5001
18016/25000 [====================>.........] - ETA: 20s - loss: 7.6598 - accuracy: 0.5004
18048/25000 [====================>.........] - ETA: 20s - loss: 7.6641 - accuracy: 0.5002
18080/25000 [====================>.........] - ETA: 20s - loss: 7.6675 - accuracy: 0.4999
18112/25000 [====================>.........] - ETA: 20s - loss: 7.6717 - accuracy: 0.4997
18144/25000 [====================>.........] - ETA: 20s - loss: 7.6734 - accuracy: 0.4996
18176/25000 [====================>.........] - ETA: 20s - loss: 7.6717 - accuracy: 0.4997
18208/25000 [====================>.........] - ETA: 20s - loss: 7.6717 - accuracy: 0.4997
18240/25000 [====================>.........] - ETA: 20s - loss: 7.6750 - accuracy: 0.4995
18272/25000 [====================>.........] - ETA: 20s - loss: 7.6759 - accuracy: 0.4994
18304/25000 [====================>.........] - ETA: 19s - loss: 7.6750 - accuracy: 0.4995
18336/25000 [=====================>........] - ETA: 19s - loss: 7.6758 - accuracy: 0.4994
18368/25000 [=====================>........] - ETA: 19s - loss: 7.6725 - accuracy: 0.4996
18400/25000 [=====================>........] - ETA: 19s - loss: 7.6691 - accuracy: 0.4998
18432/25000 [=====================>........] - ETA: 19s - loss: 7.6625 - accuracy: 0.5003
18464/25000 [=====================>........] - ETA: 19s - loss: 7.6625 - accuracy: 0.5003
18496/25000 [=====================>........] - ETA: 19s - loss: 7.6641 - accuracy: 0.5002
18528/25000 [=====================>........] - ETA: 19s - loss: 7.6658 - accuracy: 0.5001
18560/25000 [=====================>........] - ETA: 19s - loss: 7.6575 - accuracy: 0.5006
18592/25000 [=====================>........] - ETA: 19s - loss: 7.6534 - accuracy: 0.5009
18624/25000 [=====================>........] - ETA: 18s - loss: 7.6543 - accuracy: 0.5008
18656/25000 [=====================>........] - ETA: 18s - loss: 7.6584 - accuracy: 0.5005
18688/25000 [=====================>........] - ETA: 18s - loss: 7.6609 - accuracy: 0.5004
18720/25000 [=====================>........] - ETA: 18s - loss: 7.6601 - accuracy: 0.5004
18752/25000 [=====================>........] - ETA: 18s - loss: 7.6568 - accuracy: 0.5006
18784/25000 [=====================>........] - ETA: 18s - loss: 7.6593 - accuracy: 0.5005
18816/25000 [=====================>........] - ETA: 18s - loss: 7.6593 - accuracy: 0.5005
18848/25000 [=====================>........] - ETA: 18s - loss: 7.6528 - accuracy: 0.5009
18880/25000 [=====================>........] - ETA: 18s - loss: 7.6496 - accuracy: 0.5011
18912/25000 [=====================>........] - ETA: 18s - loss: 7.6504 - accuracy: 0.5011
18944/25000 [=====================>........] - ETA: 18s - loss: 7.6545 - accuracy: 0.5008
18976/25000 [=====================>........] - ETA: 17s - loss: 7.6577 - accuracy: 0.5006
19008/25000 [=====================>........] - ETA: 17s - loss: 7.6553 - accuracy: 0.5007
19040/25000 [=====================>........] - ETA: 17s - loss: 7.6505 - accuracy: 0.5011
19072/25000 [=====================>........] - ETA: 17s - loss: 7.6505 - accuracy: 0.5010
19104/25000 [=====================>........] - ETA: 17s - loss: 7.6546 - accuracy: 0.5008
19136/25000 [=====================>........] - ETA: 17s - loss: 7.6546 - accuracy: 0.5008
19168/25000 [======================>.......] - ETA: 17s - loss: 7.6538 - accuracy: 0.5008
19200/25000 [======================>.......] - ETA: 17s - loss: 7.6530 - accuracy: 0.5009
19232/25000 [======================>.......] - ETA: 17s - loss: 7.6547 - accuracy: 0.5008
19264/25000 [======================>.......] - ETA: 17s - loss: 7.6571 - accuracy: 0.5006
19296/25000 [======================>.......] - ETA: 16s - loss: 7.6579 - accuracy: 0.5006
19328/25000 [======================>.......] - ETA: 16s - loss: 7.6555 - accuracy: 0.5007
19360/25000 [======================>.......] - ETA: 16s - loss: 7.6571 - accuracy: 0.5006
19392/25000 [======================>.......] - ETA: 16s - loss: 7.6579 - accuracy: 0.5006
19424/25000 [======================>.......] - ETA: 16s - loss: 7.6587 - accuracy: 0.5005
19456/25000 [======================>.......] - ETA: 16s - loss: 7.6572 - accuracy: 0.5006
19488/25000 [======================>.......] - ETA: 16s - loss: 7.6588 - accuracy: 0.5005
19520/25000 [======================>.......] - ETA: 16s - loss: 7.6588 - accuracy: 0.5005
19552/25000 [======================>.......] - ETA: 16s - loss: 7.6564 - accuracy: 0.5007
19584/25000 [======================>.......] - ETA: 16s - loss: 7.6611 - accuracy: 0.5004
19616/25000 [======================>.......] - ETA: 16s - loss: 7.6611 - accuracy: 0.5004
19648/25000 [======================>.......] - ETA: 15s - loss: 7.6643 - accuracy: 0.5002
19680/25000 [======================>.......] - ETA: 15s - loss: 7.6635 - accuracy: 0.5002
19712/25000 [======================>.......] - ETA: 15s - loss: 7.6627 - accuracy: 0.5003
19744/25000 [======================>.......] - ETA: 15s - loss: 7.6635 - accuracy: 0.5002
19776/25000 [======================>.......] - ETA: 15s - loss: 7.6596 - accuracy: 0.5005
19808/25000 [======================>.......] - ETA: 15s - loss: 7.6612 - accuracy: 0.5004
19840/25000 [======================>.......] - ETA: 15s - loss: 7.6612 - accuracy: 0.5004
19872/25000 [======================>.......] - ETA: 15s - loss: 7.6597 - accuracy: 0.5005
19904/25000 [======================>.......] - ETA: 15s - loss: 7.6628 - accuracy: 0.5003
19936/25000 [======================>.......] - ETA: 15s - loss: 7.6597 - accuracy: 0.5005
19968/25000 [======================>.......] - ETA: 14s - loss: 7.6605 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 14s - loss: 7.6620 - accuracy: 0.5003
20032/25000 [=======================>......] - ETA: 14s - loss: 7.6628 - accuracy: 0.5002
20064/25000 [=======================>......] - ETA: 14s - loss: 7.6651 - accuracy: 0.5001
20096/25000 [=======================>......] - ETA: 14s - loss: 7.6628 - accuracy: 0.5002
20128/25000 [=======================>......] - ETA: 14s - loss: 7.6628 - accuracy: 0.5002
20160/25000 [=======================>......] - ETA: 14s - loss: 7.6598 - accuracy: 0.5004
20192/25000 [=======================>......] - ETA: 14s - loss: 7.6583 - accuracy: 0.5005
20224/25000 [=======================>......] - ETA: 14s - loss: 7.6583 - accuracy: 0.5005
20256/25000 [=======================>......] - ETA: 14s - loss: 7.6598 - accuracy: 0.5004
20288/25000 [=======================>......] - ETA: 14s - loss: 7.6583 - accuracy: 0.5005
20320/25000 [=======================>......] - ETA: 13s - loss: 7.6644 - accuracy: 0.5001
20352/25000 [=======================>......] - ETA: 13s - loss: 7.6644 - accuracy: 0.5001
20384/25000 [=======================>......] - ETA: 13s - loss: 7.6681 - accuracy: 0.4999
20416/25000 [=======================>......] - ETA: 13s - loss: 7.6651 - accuracy: 0.5001
20448/25000 [=======================>......] - ETA: 13s - loss: 7.6659 - accuracy: 0.5000
20480/25000 [=======================>......] - ETA: 13s - loss: 7.6621 - accuracy: 0.5003
20512/25000 [=======================>......] - ETA: 13s - loss: 7.6629 - accuracy: 0.5002
20544/25000 [=======================>......] - ETA: 13s - loss: 7.6621 - accuracy: 0.5003
20576/25000 [=======================>......] - ETA: 13s - loss: 7.6599 - accuracy: 0.5004
20608/25000 [=======================>......] - ETA: 13s - loss: 7.6599 - accuracy: 0.5004
20640/25000 [=======================>......] - ETA: 12s - loss: 7.6614 - accuracy: 0.5003
20672/25000 [=======================>......] - ETA: 12s - loss: 7.6644 - accuracy: 0.5001
20704/25000 [=======================>......] - ETA: 12s - loss: 7.6622 - accuracy: 0.5003
20736/25000 [=======================>......] - ETA: 12s - loss: 7.6659 - accuracy: 0.5000
20768/25000 [=======================>......] - ETA: 12s - loss: 7.6659 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 12s - loss: 7.6644 - accuracy: 0.5001
20832/25000 [=======================>......] - ETA: 12s - loss: 7.6629 - accuracy: 0.5002
20864/25000 [========================>.....] - ETA: 12s - loss: 7.6615 - accuracy: 0.5003
20896/25000 [========================>.....] - ETA: 12s - loss: 7.6644 - accuracy: 0.5001
20928/25000 [========================>.....] - ETA: 12s - loss: 7.6630 - accuracy: 0.5002
20960/25000 [========================>.....] - ETA: 12s - loss: 7.6622 - accuracy: 0.5003
20992/25000 [========================>.....] - ETA: 11s - loss: 7.6615 - accuracy: 0.5003
21024/25000 [========================>.....] - ETA: 11s - loss: 7.6615 - accuracy: 0.5003
21056/25000 [========================>.....] - ETA: 11s - loss: 7.6630 - accuracy: 0.5002
21088/25000 [========================>.....] - ETA: 11s - loss: 7.6601 - accuracy: 0.5004
21120/25000 [========================>.....] - ETA: 11s - loss: 7.6601 - accuracy: 0.5004
21152/25000 [========================>.....] - ETA: 11s - loss: 7.6601 - accuracy: 0.5004
21184/25000 [========================>.....] - ETA: 11s - loss: 7.6616 - accuracy: 0.5003
21216/25000 [========================>.....] - ETA: 11s - loss: 7.6601 - accuracy: 0.5004
21248/25000 [========================>.....] - ETA: 11s - loss: 7.6608 - accuracy: 0.5004
21280/25000 [========================>.....] - ETA: 11s - loss: 7.6623 - accuracy: 0.5003
21312/25000 [========================>.....] - ETA: 10s - loss: 7.6630 - accuracy: 0.5002
21344/25000 [========================>.....] - ETA: 10s - loss: 7.6623 - accuracy: 0.5003
21376/25000 [========================>.....] - ETA: 10s - loss: 7.6623 - accuracy: 0.5003
21408/25000 [========================>.....] - ETA: 10s - loss: 7.6623 - accuracy: 0.5003
21440/25000 [========================>.....] - ETA: 10s - loss: 7.6645 - accuracy: 0.5001
21472/25000 [========================>.....] - ETA: 10s - loss: 7.6645 - accuracy: 0.5001
21504/25000 [========================>.....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
21536/25000 [========================>.....] - ETA: 10s - loss: 7.6638 - accuracy: 0.5002
21568/25000 [========================>.....] - ETA: 10s - loss: 7.6616 - accuracy: 0.5003
21600/25000 [========================>.....] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
21632/25000 [========================>.....] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
21664/25000 [========================>.....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001 
21696/25000 [=========================>....] - ETA: 9s - loss: 7.6631 - accuracy: 0.5002
21728/25000 [=========================>....] - ETA: 9s - loss: 7.6638 - accuracy: 0.5002
21760/25000 [=========================>....] - ETA: 9s - loss: 7.6638 - accuracy: 0.5002
21792/25000 [=========================>....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21824/25000 [=========================>....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
21856/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
21888/25000 [=========================>....] - ETA: 9s - loss: 7.6715 - accuracy: 0.4997
21920/25000 [=========================>....] - ETA: 9s - loss: 7.6722 - accuracy: 0.4996
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6701 - accuracy: 0.4998
21984/25000 [=========================>....] - ETA: 8s - loss: 7.6701 - accuracy: 0.4998
22016/25000 [=========================>....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
22048/25000 [=========================>....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
22080/25000 [=========================>....] - ETA: 8s - loss: 7.6687 - accuracy: 0.4999
22112/25000 [=========================>....] - ETA: 8s - loss: 7.6701 - accuracy: 0.4998
22144/25000 [=========================>....] - ETA: 8s - loss: 7.6729 - accuracy: 0.4996
22176/25000 [=========================>....] - ETA: 8s - loss: 7.6756 - accuracy: 0.4994
22208/25000 [=========================>....] - ETA: 8s - loss: 7.6763 - accuracy: 0.4994
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6742 - accuracy: 0.4995
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6742 - accuracy: 0.4995
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6721 - accuracy: 0.4996
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6707 - accuracy: 0.4997
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6687 - accuracy: 0.4999
22400/25000 [=========================>....] - ETA: 7s - loss: 7.6700 - accuracy: 0.4998
22432/25000 [=========================>....] - ETA: 7s - loss: 7.6728 - accuracy: 0.4996
22464/25000 [=========================>....] - ETA: 7s - loss: 7.6721 - accuracy: 0.4996
22496/25000 [=========================>....] - ETA: 7s - loss: 7.6755 - accuracy: 0.4994
22528/25000 [==========================>...] - ETA: 7s - loss: 7.6768 - accuracy: 0.4993
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6775 - accuracy: 0.4993
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6795 - accuracy: 0.4992
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6822 - accuracy: 0.4990
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6815 - accuracy: 0.4990
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6801 - accuracy: 0.4991
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6794 - accuracy: 0.4992
22752/25000 [==========================>...] - ETA: 6s - loss: 7.6747 - accuracy: 0.4995
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6727 - accuracy: 0.4996
22816/25000 [==========================>...] - ETA: 6s - loss: 7.6740 - accuracy: 0.4995
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6760 - accuracy: 0.4994
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6747 - accuracy: 0.4995
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6773 - accuracy: 0.4993
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6780 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6780 - accuracy: 0.4993
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6799 - accuracy: 0.4991
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6766 - accuracy: 0.4993
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6773 - accuracy: 0.4993
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6746 - accuracy: 0.4995
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6759 - accuracy: 0.4994
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6785 - accuracy: 0.4992
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6759 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6778 - accuracy: 0.4993
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6772 - accuracy: 0.4993
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6791 - accuracy: 0.4992
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6752 - accuracy: 0.4994
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6738 - accuracy: 0.4995
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6751 - accuracy: 0.4994
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6660 - accuracy: 0.5000
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6608 - accuracy: 0.5004
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6589 - accuracy: 0.5005
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6576 - accuracy: 0.5006
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6570 - accuracy: 0.5006
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6570 - accuracy: 0.5006
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6570 - accuracy: 0.5006
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6538 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6520 - accuracy: 0.5010
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6495 - accuracy: 0.5011
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6501 - accuracy: 0.5011
24192/25000 [============================>.] - ETA: 2s - loss: 7.6514 - accuracy: 0.5010
24224/25000 [============================>.] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
24256/25000 [============================>.] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
24288/25000 [============================>.] - ETA: 2s - loss: 7.6553 - accuracy: 0.5007
24320/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24352/25000 [============================>.] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24384/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
24416/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24448/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24480/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24576/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24704/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24736/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 90s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

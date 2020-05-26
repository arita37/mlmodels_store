
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
	Data preprocessing and feature engineering runtime = 0.33s ...
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
 40%|████      | 2/5 [01:01<01:31, 30.53s/it] 40%|████      | 2/5 [01:01<01:31, 30.54s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3293099152643445, 'embedding_size_factor': 0.6751961132284172, 'layers.choice': 3, 'learning_rate': 0.003043697839083899, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 8.348614593383656e-09} and reward: 0.3842
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\x13i\xe5\x13\xc4\xdeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x9b4\xe1\x16y\xc6X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?h\xef\x18\xd5\xb0MCX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>A\xed\xb3\x0c\x93\x17\xafu.' and reward: 0.3842
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\x13i\xe5\x13\xc4\xdeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x9b4\xe1\x16y\xc6X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?h\xef\x18\xd5\xb0MCX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>A\xed\xb3\x0c\x93\x17\xafu.' and reward: 0.3842
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 126.36027979850769
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.67s of the -8.46s of remaining time.
Ensemble size: 41
Ensemble weights: 
[0.68292683 0.31707317]
	0.3886	 = Validation accuracy score
	1.02s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 129.53s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
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
 2023424/17464789 [==>...........................] - ETA: 0s
 5398528/17464789 [========>.....................] - ETA: 0s
 8642560/17464789 [=============>................] - ETA: 0s
12050432/17464789 [===================>..........] - ETA: 0s
15425536/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-26 05:21:42.491531: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 05:21:42.496831: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-26 05:21:42.497079: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557de696c4e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 05:21:42.497098: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:22 - loss: 7.1875 - accuracy: 0.5312
   64/25000 [..............................] - ETA: 3:26 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:47 - loss: 7.6666 - accuracy: 0.5000
  128/25000 [..............................] - ETA: 2:27 - loss: 7.0677 - accuracy: 0.5391
  160/25000 [..............................] - ETA: 2:14 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 2:06 - loss: 7.1875 - accuracy: 0.5312
  224/25000 [..............................] - ETA: 2:01 - loss: 6.9821 - accuracy: 0.5446
  256/25000 [..............................] - ETA: 1:57 - loss: 7.0078 - accuracy: 0.5430
  288/25000 [..............................] - ETA: 1:54 - loss: 7.1875 - accuracy: 0.5312
  320/25000 [..............................] - ETA: 1:52 - loss: 7.1875 - accuracy: 0.5312
  352/25000 [..............................] - ETA: 1:49 - loss: 7.1439 - accuracy: 0.5341
  384/25000 [..............................] - ETA: 1:48 - loss: 7.1076 - accuracy: 0.5365
  416/25000 [..............................] - ETA: 1:46 - loss: 7.0032 - accuracy: 0.5433
  448/25000 [..............................] - ETA: 1:45 - loss: 7.0506 - accuracy: 0.5402
  480/25000 [..............................] - ETA: 1:44 - loss: 7.0277 - accuracy: 0.5417
  512/25000 [..............................] - ETA: 1:43 - loss: 7.1575 - accuracy: 0.5332
  544/25000 [..............................] - ETA: 1:42 - loss: 7.1875 - accuracy: 0.5312
  576/25000 [..............................] - ETA: 1:41 - loss: 7.2407 - accuracy: 0.5278
  608/25000 [..............................] - ETA: 1:40 - loss: 7.2379 - accuracy: 0.5280
  640/25000 [..............................] - ETA: 1:39 - loss: 7.1875 - accuracy: 0.5312
  672/25000 [..............................] - ETA: 1:39 - loss: 7.2559 - accuracy: 0.5268
  704/25000 [..............................] - ETA: 1:39 - loss: 7.1439 - accuracy: 0.5341
  736/25000 [..............................] - ETA: 1:38 - loss: 7.0416 - accuracy: 0.5408
  768/25000 [..............................] - ETA: 1:38 - loss: 7.0477 - accuracy: 0.5404
  800/25000 [..............................] - ETA: 1:37 - loss: 7.1491 - accuracy: 0.5337
  832/25000 [..............................] - ETA: 1:37 - loss: 7.2243 - accuracy: 0.5288
  864/25000 [>.............................] - ETA: 1:36 - loss: 7.1875 - accuracy: 0.5312
  896/25000 [>.............................] - ETA: 1:36 - loss: 7.1875 - accuracy: 0.5312
  928/25000 [>.............................] - ETA: 1:35 - loss: 7.1709 - accuracy: 0.5323
  960/25000 [>.............................] - ETA: 1:35 - loss: 7.1875 - accuracy: 0.5312
  992/25000 [>.............................] - ETA: 1:34 - loss: 7.2029 - accuracy: 0.5302
 1024/25000 [>.............................] - ETA: 1:34 - loss: 7.2623 - accuracy: 0.5264
 1056/25000 [>.............................] - ETA: 1:33 - loss: 7.2455 - accuracy: 0.5275
 1088/25000 [>.............................] - ETA: 1:33 - loss: 7.2861 - accuracy: 0.5248
 1120/25000 [>.............................] - ETA: 1:33 - loss: 7.2833 - accuracy: 0.5250
 1152/25000 [>.............................] - ETA: 1:32 - loss: 7.2806 - accuracy: 0.5252
 1184/25000 [>.............................] - ETA: 1:32 - loss: 7.2781 - accuracy: 0.5253
 1216/25000 [>.............................] - ETA: 1:32 - loss: 7.2883 - accuracy: 0.5247
 1248/25000 [>.............................] - ETA: 1:31 - loss: 7.3349 - accuracy: 0.5216
 1280/25000 [>.............................] - ETA: 1:31 - loss: 7.3552 - accuracy: 0.5203
 1312/25000 [>.............................] - ETA: 1:31 - loss: 7.3861 - accuracy: 0.5183
 1344/25000 [>.............................] - ETA: 1:31 - loss: 7.4384 - accuracy: 0.5149
 1376/25000 [>.............................] - ETA: 1:30 - loss: 7.3992 - accuracy: 0.5174
 1408/25000 [>.............................] - ETA: 1:30 - loss: 7.4161 - accuracy: 0.5163
 1440/25000 [>.............................] - ETA: 1:30 - loss: 7.3898 - accuracy: 0.5181
 1472/25000 [>.............................] - ETA: 1:30 - loss: 7.3854 - accuracy: 0.5183
 1504/25000 [>.............................] - ETA: 1:29 - loss: 7.3608 - accuracy: 0.5199
 1536/25000 [>.............................] - ETA: 1:29 - loss: 7.4171 - accuracy: 0.5163
 1568/25000 [>.............................] - ETA: 1:29 - loss: 7.4515 - accuracy: 0.5140
 1600/25000 [>.............................] - ETA: 1:29 - loss: 7.4654 - accuracy: 0.5131
 1632/25000 [>.............................] - ETA: 1:29 - loss: 7.4505 - accuracy: 0.5141
 1664/25000 [>.............................] - ETA: 1:29 - loss: 7.5008 - accuracy: 0.5108
 1696/25000 [=>............................] - ETA: 1:28 - loss: 7.5310 - accuracy: 0.5088
 1728/25000 [=>............................] - ETA: 1:28 - loss: 7.5513 - accuracy: 0.5075
 1760/25000 [=>............................] - ETA: 1:28 - loss: 7.5272 - accuracy: 0.5091
 1792/25000 [=>............................] - ETA: 1:28 - loss: 7.5212 - accuracy: 0.5095
 1824/25000 [=>............................] - ETA: 1:28 - loss: 7.5069 - accuracy: 0.5104
 1856/25000 [=>............................] - ETA: 1:28 - loss: 7.5427 - accuracy: 0.5081
 1888/25000 [=>............................] - ETA: 1:28 - loss: 7.5367 - accuracy: 0.5085
 1920/25000 [=>............................] - ETA: 1:27 - loss: 7.5388 - accuracy: 0.5083
 1952/25000 [=>............................] - ETA: 1:27 - loss: 7.5252 - accuracy: 0.5092
 1984/25000 [=>............................] - ETA: 1:27 - loss: 7.5043 - accuracy: 0.5106
 2016/25000 [=>............................] - ETA: 1:27 - loss: 7.4689 - accuracy: 0.5129
 2048/25000 [=>............................] - ETA: 1:27 - loss: 7.4495 - accuracy: 0.5142
 2080/25000 [=>............................] - ETA: 1:27 - loss: 7.4307 - accuracy: 0.5154
 2112/25000 [=>............................] - ETA: 1:26 - loss: 7.4561 - accuracy: 0.5137
 2144/25000 [=>............................] - ETA: 1:26 - loss: 7.4449 - accuracy: 0.5145
 2176/25000 [=>............................] - ETA: 1:26 - loss: 7.4270 - accuracy: 0.5156
 2208/25000 [=>............................] - ETA: 1:26 - loss: 7.4236 - accuracy: 0.5159
 2240/25000 [=>............................] - ETA: 1:26 - loss: 7.4613 - accuracy: 0.5134
 2272/25000 [=>............................] - ETA: 1:26 - loss: 7.4507 - accuracy: 0.5141
 2304/25000 [=>............................] - ETA: 1:25 - loss: 7.4803 - accuracy: 0.5122
 2336/25000 [=>............................] - ETA: 1:25 - loss: 7.4960 - accuracy: 0.5111
 2368/25000 [=>............................] - ETA: 1:25 - loss: 7.5047 - accuracy: 0.5106
 2400/25000 [=>............................] - ETA: 1:25 - loss: 7.5197 - accuracy: 0.5096
 2432/25000 [=>............................] - ETA: 1:25 - loss: 7.4901 - accuracy: 0.5115
 2464/25000 [=>............................] - ETA: 1:25 - loss: 7.5048 - accuracy: 0.5106
 2496/25000 [=>............................] - ETA: 1:25 - loss: 7.5315 - accuracy: 0.5088
 2528/25000 [==>...........................] - ETA: 1:24 - loss: 7.5271 - accuracy: 0.5091
 2560/25000 [==>...........................] - ETA: 1:24 - loss: 7.5049 - accuracy: 0.5105
 2592/25000 [==>...........................] - ETA: 1:24 - loss: 7.5010 - accuracy: 0.5108
 2624/25000 [==>...........................] - ETA: 1:24 - loss: 7.5439 - accuracy: 0.5080
 2656/25000 [==>...........................] - ETA: 1:24 - loss: 7.5165 - accuracy: 0.5098
 2688/25000 [==>...........................] - ETA: 1:24 - loss: 7.4784 - accuracy: 0.5123
 2720/25000 [==>...........................] - ETA: 1:24 - loss: 7.4355 - accuracy: 0.5151
 2752/25000 [==>...........................] - ETA: 1:23 - loss: 7.4438 - accuracy: 0.5145
 2784/25000 [==>...........................] - ETA: 1:23 - loss: 7.4518 - accuracy: 0.5140
 2816/25000 [==>...........................] - ETA: 1:23 - loss: 7.4325 - accuracy: 0.5153
 2848/25000 [==>...........................] - ETA: 1:23 - loss: 7.4566 - accuracy: 0.5137
 2880/25000 [==>...........................] - ETA: 1:23 - loss: 7.4750 - accuracy: 0.5125
 2912/25000 [==>...........................] - ETA: 1:23 - loss: 7.4823 - accuracy: 0.5120
 2944/25000 [==>...........................] - ETA: 1:22 - loss: 7.5000 - accuracy: 0.5109
 2976/25000 [==>...........................] - ETA: 1:22 - loss: 7.4914 - accuracy: 0.5114
 3008/25000 [==>...........................] - ETA: 1:22 - loss: 7.4933 - accuracy: 0.5113
 3040/25000 [==>...........................] - ETA: 1:22 - loss: 7.4750 - accuracy: 0.5125
 3072/25000 [==>...........................] - ETA: 1:22 - loss: 7.4969 - accuracy: 0.5111
 3104/25000 [==>...........................] - ETA: 1:22 - loss: 7.5036 - accuracy: 0.5106
 3136/25000 [==>...........................] - ETA: 1:21 - loss: 7.5053 - accuracy: 0.5105
 3168/25000 [==>...........................] - ETA: 1:21 - loss: 7.5021 - accuracy: 0.5107
 3200/25000 [==>...........................] - ETA: 1:21 - loss: 7.5181 - accuracy: 0.5097
 3232/25000 [==>...........................] - ETA: 1:21 - loss: 7.4911 - accuracy: 0.5114
 3264/25000 [==>...........................] - ETA: 1:21 - loss: 7.4928 - accuracy: 0.5113
 3296/25000 [==>...........................] - ETA: 1:21 - loss: 7.5131 - accuracy: 0.5100
 3328/25000 [==>...........................] - ETA: 1:21 - loss: 7.5192 - accuracy: 0.5096
 3360/25000 [===>..........................] - ETA: 1:20 - loss: 7.4841 - accuracy: 0.5119
 3392/25000 [===>..........................] - ETA: 1:20 - loss: 7.4858 - accuracy: 0.5118
 3424/25000 [===>..........................] - ETA: 1:20 - loss: 7.4785 - accuracy: 0.5123
 3456/25000 [===>..........................] - ETA: 1:20 - loss: 7.4847 - accuracy: 0.5119
 3488/25000 [===>..........................] - ETA: 1:20 - loss: 7.4776 - accuracy: 0.5123
 3520/25000 [===>..........................] - ETA: 1:20 - loss: 7.4880 - accuracy: 0.5116
 3552/25000 [===>..........................] - ETA: 1:20 - loss: 7.4983 - accuracy: 0.5110
 3584/25000 [===>..........................] - ETA: 1:19 - loss: 7.5169 - accuracy: 0.5098
 3616/25000 [===>..........................] - ETA: 1:19 - loss: 7.5140 - accuracy: 0.5100
 3648/25000 [===>..........................] - ETA: 1:19 - loss: 7.5153 - accuracy: 0.5099
 3680/25000 [===>..........................] - ETA: 1:19 - loss: 7.5208 - accuracy: 0.5095
 3712/25000 [===>..........................] - ETA: 1:19 - loss: 7.5303 - accuracy: 0.5089
 3744/25000 [===>..........................] - ETA: 1:19 - loss: 7.5315 - accuracy: 0.5088
 3776/25000 [===>..........................] - ETA: 1:19 - loss: 7.5326 - accuracy: 0.5087
 3808/25000 [===>..........................] - ETA: 1:19 - loss: 7.5297 - accuracy: 0.5089
 3840/25000 [===>..........................] - ETA: 1:18 - loss: 7.5348 - accuracy: 0.5086
 3872/25000 [===>..........................] - ETA: 1:18 - loss: 7.5359 - accuracy: 0.5085
 3904/25000 [===>..........................] - ETA: 1:18 - loss: 7.5409 - accuracy: 0.5082
 3936/25000 [===>..........................] - ETA: 1:18 - loss: 7.5381 - accuracy: 0.5084
 3968/25000 [===>..........................] - ETA: 1:18 - loss: 7.5468 - accuracy: 0.5078
 4000/25000 [===>..........................] - ETA: 1:18 - loss: 7.5516 - accuracy: 0.5075
 4032/25000 [===>..........................] - ETA: 1:17 - loss: 7.5601 - accuracy: 0.5069
 4064/25000 [===>..........................] - ETA: 1:17 - loss: 7.5459 - accuracy: 0.5079
 4096/25000 [===>..........................] - ETA: 1:17 - loss: 7.5543 - accuracy: 0.5073
 4128/25000 [===>..........................] - ETA: 1:17 - loss: 7.5515 - accuracy: 0.5075
 4160/25000 [===>..........................] - ETA: 1:17 - loss: 7.5413 - accuracy: 0.5082
 4192/25000 [====>.........................] - ETA: 1:17 - loss: 7.5386 - accuracy: 0.5083
 4224/25000 [====>.........................] - ETA: 1:17 - loss: 7.5359 - accuracy: 0.5085
 4256/25000 [====>.........................] - ETA: 1:17 - loss: 7.5477 - accuracy: 0.5078
 4288/25000 [====>.........................] - ETA: 1:16 - loss: 7.5450 - accuracy: 0.5079
 4320/25000 [====>.........................] - ETA: 1:16 - loss: 7.5388 - accuracy: 0.5083
 4352/25000 [====>.........................] - ETA: 1:16 - loss: 7.5433 - accuracy: 0.5080
 4384/25000 [====>.........................] - ETA: 1:16 - loss: 7.5337 - accuracy: 0.5087
 4416/25000 [====>.........................] - ETA: 1:16 - loss: 7.5520 - accuracy: 0.5075
 4448/25000 [====>.........................] - ETA: 1:16 - loss: 7.5632 - accuracy: 0.5067
 4480/25000 [====>.........................] - ETA: 1:16 - loss: 7.5742 - accuracy: 0.5060
 4512/25000 [====>.........................] - ETA: 1:15 - loss: 7.5851 - accuracy: 0.5053
 4544/25000 [====>.........................] - ETA: 1:15 - loss: 7.5755 - accuracy: 0.5059
 4576/25000 [====>.........................] - ETA: 1:15 - loss: 7.5828 - accuracy: 0.5055
 4608/25000 [====>.........................] - ETA: 1:15 - loss: 7.5901 - accuracy: 0.5050
 4640/25000 [====>.........................] - ETA: 1:15 - loss: 7.6038 - accuracy: 0.5041
 4672/25000 [====>.........................] - ETA: 1:15 - loss: 7.5977 - accuracy: 0.5045
 4704/25000 [====>.........................] - ETA: 1:15 - loss: 7.5884 - accuracy: 0.5051
 4736/25000 [====>.........................] - ETA: 1:15 - loss: 7.5986 - accuracy: 0.5044
 4768/25000 [====>.........................] - ETA: 1:14 - loss: 7.5927 - accuracy: 0.5048
 4800/25000 [====>.........................] - ETA: 1:14 - loss: 7.5931 - accuracy: 0.5048
 4832/25000 [====>.........................] - ETA: 1:14 - loss: 7.5873 - accuracy: 0.5052
 4864/25000 [====>.........................] - ETA: 1:14 - loss: 7.5784 - accuracy: 0.5058
 4896/25000 [====>.........................] - ETA: 1:14 - loss: 7.5946 - accuracy: 0.5047
 4928/25000 [====>.........................] - ETA: 1:14 - loss: 7.5982 - accuracy: 0.5045
 4960/25000 [====>.........................] - ETA: 1:14 - loss: 7.6079 - accuracy: 0.5038
 4992/25000 [====>.........................] - ETA: 1:14 - loss: 7.6113 - accuracy: 0.5036
 5024/25000 [=====>........................] - ETA: 1:13 - loss: 7.6086 - accuracy: 0.5038
 5056/25000 [=====>........................] - ETA: 1:13 - loss: 7.6211 - accuracy: 0.5030
 5088/25000 [=====>........................] - ETA: 1:13 - loss: 7.6124 - accuracy: 0.5035
 5120/25000 [=====>........................] - ETA: 1:13 - loss: 7.5947 - accuracy: 0.5047
 5152/25000 [=====>........................] - ETA: 1:13 - loss: 7.5803 - accuracy: 0.5056
 5184/25000 [=====>........................] - ETA: 1:13 - loss: 7.5720 - accuracy: 0.5062
 5216/25000 [=====>........................] - ETA: 1:13 - loss: 7.5696 - accuracy: 0.5063
 5248/25000 [=====>........................] - ETA: 1:13 - loss: 7.5614 - accuracy: 0.5069
 5280/25000 [=====>........................] - ETA: 1:12 - loss: 7.5563 - accuracy: 0.5072
 5312/25000 [=====>........................] - ETA: 1:12 - loss: 7.5627 - accuracy: 0.5068
 5344/25000 [=====>........................] - ETA: 1:12 - loss: 7.5518 - accuracy: 0.5075
 5376/25000 [=====>........................] - ETA: 1:12 - loss: 7.5668 - accuracy: 0.5065
 5408/25000 [=====>........................] - ETA: 1:12 - loss: 7.5589 - accuracy: 0.5070
 5440/25000 [=====>........................] - ETA: 1:12 - loss: 7.5567 - accuracy: 0.5072
 5472/25000 [=====>........................] - ETA: 1:12 - loss: 7.5517 - accuracy: 0.5075
 5504/25000 [=====>........................] - ETA: 1:12 - loss: 7.5552 - accuracy: 0.5073
 5536/25000 [=====>........................] - ETA: 1:11 - loss: 7.5586 - accuracy: 0.5070
 5568/25000 [=====>........................] - ETA: 1:11 - loss: 7.5592 - accuracy: 0.5070
 5600/25000 [=====>........................] - ETA: 1:11 - loss: 7.5653 - accuracy: 0.5066
 5632/25000 [=====>........................] - ETA: 1:11 - loss: 7.5741 - accuracy: 0.5060
 5664/25000 [=====>........................] - ETA: 1:11 - loss: 7.5719 - accuracy: 0.5062
 5696/25000 [=====>........................] - ETA: 1:11 - loss: 7.5643 - accuracy: 0.5067
 5728/25000 [=====>........................] - ETA: 1:11 - loss: 7.5622 - accuracy: 0.5068
 5760/25000 [=====>........................] - ETA: 1:11 - loss: 7.5655 - accuracy: 0.5066
 5792/25000 [=====>........................] - ETA: 1:10 - loss: 7.5581 - accuracy: 0.5071
 5824/25000 [=====>........................] - ETA: 1:10 - loss: 7.5560 - accuracy: 0.5072
 5856/25000 [======>.......................] - ETA: 1:10 - loss: 7.5566 - accuracy: 0.5072
 5888/25000 [======>.......................] - ETA: 1:10 - loss: 7.5494 - accuracy: 0.5076
 5920/25000 [======>.......................] - ETA: 1:10 - loss: 7.5423 - accuracy: 0.5081
 5952/25000 [======>.......................] - ETA: 1:10 - loss: 7.5249 - accuracy: 0.5092
 5984/25000 [======>.......................] - ETA: 1:10 - loss: 7.5308 - accuracy: 0.5089
 6016/25000 [======>.......................] - ETA: 1:10 - loss: 7.5264 - accuracy: 0.5091
 6048/25000 [======>.......................] - ETA: 1:10 - loss: 7.5272 - accuracy: 0.5091
 6080/25000 [======>.......................] - ETA: 1:09 - loss: 7.5279 - accuracy: 0.5090
 6112/25000 [======>.......................] - ETA: 1:09 - loss: 7.5337 - accuracy: 0.5087
 6144/25000 [======>.......................] - ETA: 1:09 - loss: 7.5344 - accuracy: 0.5086
 6176/25000 [======>.......................] - ETA: 1:09 - loss: 7.5301 - accuracy: 0.5089
 6208/25000 [======>.......................] - ETA: 1:09 - loss: 7.5283 - accuracy: 0.5090
 6240/25000 [======>.......................] - ETA: 1:09 - loss: 7.5339 - accuracy: 0.5087
 6272/25000 [======>.......................] - ETA: 1:09 - loss: 7.5273 - accuracy: 0.5091
 6304/25000 [======>.......................] - ETA: 1:09 - loss: 7.5255 - accuracy: 0.5092
 6336/25000 [======>.......................] - ETA: 1:09 - loss: 7.5287 - accuracy: 0.5090
 6368/25000 [======>.......................] - ETA: 1:08 - loss: 7.5366 - accuracy: 0.5085
 6400/25000 [======>.......................] - ETA: 1:08 - loss: 7.5468 - accuracy: 0.5078
 6432/25000 [======>.......................] - ETA: 1:08 - loss: 7.5522 - accuracy: 0.5075
 6464/25000 [======>.......................] - ETA: 1:08 - loss: 7.5504 - accuracy: 0.5076
 6496/25000 [======>.......................] - ETA: 1:08 - loss: 7.5557 - accuracy: 0.5072
 6528/25000 [======>.......................] - ETA: 1:08 - loss: 7.5562 - accuracy: 0.5072
 6560/25000 [======>.......................] - ETA: 1:08 - loss: 7.5638 - accuracy: 0.5067
 6592/25000 [======>.......................] - ETA: 1:08 - loss: 7.5596 - accuracy: 0.5070
 6624/25000 [======>.......................] - ETA: 1:08 - loss: 7.5578 - accuracy: 0.5071
 6656/25000 [======>.......................] - ETA: 1:07 - loss: 7.5722 - accuracy: 0.5062
 6688/25000 [=======>......................] - ETA: 1:07 - loss: 7.5703 - accuracy: 0.5063
 6720/25000 [=======>......................] - ETA: 1:07 - loss: 7.5776 - accuracy: 0.5058
 6752/25000 [=======>......................] - ETA: 1:07 - loss: 7.5758 - accuracy: 0.5059
 6784/25000 [=======>......................] - ETA: 1:07 - loss: 7.5717 - accuracy: 0.5062
 6816/25000 [=======>......................] - ETA: 1:07 - loss: 7.5789 - accuracy: 0.5057
 6848/25000 [=======>......................] - ETA: 1:07 - loss: 7.5703 - accuracy: 0.5063
 6880/25000 [=======>......................] - ETA: 1:07 - loss: 7.5708 - accuracy: 0.5063
 6912/25000 [=======>......................] - ETA: 1:07 - loss: 7.5690 - accuracy: 0.5064
 6944/25000 [=======>......................] - ETA: 1:06 - loss: 7.5650 - accuracy: 0.5066
 6976/25000 [=======>......................] - ETA: 1:06 - loss: 7.5567 - accuracy: 0.5072
 7008/25000 [=======>......................] - ETA: 1:06 - loss: 7.5682 - accuracy: 0.5064
 7040/25000 [=======>......................] - ETA: 1:06 - loss: 7.5751 - accuracy: 0.5060
 7072/25000 [=======>......................] - ETA: 1:06 - loss: 7.5799 - accuracy: 0.5057
 7104/25000 [=======>......................] - ETA: 1:06 - loss: 7.5889 - accuracy: 0.5051
 7136/25000 [=======>......................] - ETA: 1:06 - loss: 7.5871 - accuracy: 0.5052
 7168/25000 [=======>......................] - ETA: 1:06 - loss: 7.5875 - accuracy: 0.5052
 7200/25000 [=======>......................] - ETA: 1:05 - loss: 7.5836 - accuracy: 0.5054
 7232/25000 [=======>......................] - ETA: 1:05 - loss: 7.5882 - accuracy: 0.5051
 7264/25000 [=======>......................] - ETA: 1:05 - loss: 7.5801 - accuracy: 0.5056
 7296/25000 [=======>......................] - ETA: 1:05 - loss: 7.5741 - accuracy: 0.5060
 7328/25000 [=======>......................] - ETA: 1:05 - loss: 7.5746 - accuracy: 0.5060
 7360/25000 [=======>......................] - ETA: 1:05 - loss: 7.5687 - accuracy: 0.5064
 7392/25000 [=======>......................] - ETA: 1:05 - loss: 7.5691 - accuracy: 0.5064
 7424/25000 [=======>......................] - ETA: 1:05 - loss: 7.5757 - accuracy: 0.5059
 7456/25000 [=======>......................] - ETA: 1:05 - loss: 7.5741 - accuracy: 0.5060
 7488/25000 [=======>......................] - ETA: 1:04 - loss: 7.5765 - accuracy: 0.5059
 7520/25000 [========>.....................] - ETA: 1:04 - loss: 7.5769 - accuracy: 0.5059
 7552/25000 [========>.....................] - ETA: 1:04 - loss: 7.5732 - accuracy: 0.5061
 7584/25000 [========>.....................] - ETA: 1:04 - loss: 7.5777 - accuracy: 0.5058
 7616/25000 [========>.....................] - ETA: 1:04 - loss: 7.5800 - accuracy: 0.5056
 7648/25000 [========>.....................] - ETA: 1:04 - loss: 7.5804 - accuracy: 0.5056
 7680/25000 [========>.....................] - ETA: 1:04 - loss: 7.5967 - accuracy: 0.5046
 7712/25000 [========>.....................] - ETA: 1:04 - loss: 7.5851 - accuracy: 0.5053
 7744/25000 [========>.....................] - ETA: 1:04 - loss: 7.5874 - accuracy: 0.5052
 7776/25000 [========>.....................] - ETA: 1:03 - loss: 7.5838 - accuracy: 0.5054
 7808/25000 [========>.....................] - ETA: 1:03 - loss: 7.5861 - accuracy: 0.5053
 7840/25000 [========>.....................] - ETA: 1:03 - loss: 7.5923 - accuracy: 0.5048
 7872/25000 [========>.....................] - ETA: 1:03 - loss: 7.5926 - accuracy: 0.5048
 7904/25000 [========>.....................] - ETA: 1:03 - loss: 7.5890 - accuracy: 0.5051
 7936/25000 [========>.....................] - ETA: 1:03 - loss: 7.5893 - accuracy: 0.5050
 7968/25000 [========>.....................] - ETA: 1:03 - loss: 7.5916 - accuracy: 0.5049
 8000/25000 [========>.....................] - ETA: 1:03 - loss: 7.5938 - accuracy: 0.5048
 8032/25000 [========>.....................] - ETA: 1:02 - loss: 7.5979 - accuracy: 0.5045
 8064/25000 [========>.....................] - ETA: 1:02 - loss: 7.5963 - accuracy: 0.5046
 8096/25000 [========>.....................] - ETA: 1:02 - loss: 7.5871 - accuracy: 0.5052
 8128/25000 [========>.....................] - ETA: 1:02 - loss: 7.5836 - accuracy: 0.5054
 8160/25000 [========>.....................] - ETA: 1:02 - loss: 7.5802 - accuracy: 0.5056
 8192/25000 [========>.....................] - ETA: 1:02 - loss: 7.5861 - accuracy: 0.5052
 8224/25000 [========>.....................] - ETA: 1:02 - loss: 7.5864 - accuracy: 0.5052
 8256/25000 [========>.....................] - ETA: 1:02 - loss: 7.5849 - accuracy: 0.5053
 8288/25000 [========>.....................] - ETA: 1:02 - loss: 7.5741 - accuracy: 0.5060
 8320/25000 [========>.....................] - ETA: 1:01 - loss: 7.5708 - accuracy: 0.5063
 8352/25000 [=========>....................] - ETA: 1:01 - loss: 7.5730 - accuracy: 0.5061
 8384/25000 [=========>....................] - ETA: 1:01 - loss: 7.5715 - accuracy: 0.5062
 8416/25000 [=========>....................] - ETA: 1:01 - loss: 7.5846 - accuracy: 0.5053
 8448/25000 [=========>....................] - ETA: 1:01 - loss: 7.5904 - accuracy: 0.5050
 8480/25000 [=========>....................] - ETA: 1:01 - loss: 7.5871 - accuracy: 0.5052
 8512/25000 [=========>....................] - ETA: 1:01 - loss: 7.5928 - accuracy: 0.5048
 8544/25000 [=========>....................] - ETA: 1:01 - loss: 7.6020 - accuracy: 0.5042
 8576/25000 [=========>....................] - ETA: 1:01 - loss: 7.6005 - accuracy: 0.5043
 8608/25000 [=========>....................] - ETA: 1:00 - loss: 7.6007 - accuracy: 0.5043
 8640/25000 [=========>....................] - ETA: 1:00 - loss: 7.5921 - accuracy: 0.5049
 8672/25000 [=========>....................] - ETA: 1:00 - loss: 7.5959 - accuracy: 0.5046
 8704/25000 [=========>....................] - ETA: 1:00 - loss: 7.6014 - accuracy: 0.5043
 8736/25000 [=========>....................] - ETA: 1:00 - loss: 7.6034 - accuracy: 0.5041
 8768/25000 [=========>....................] - ETA: 1:00 - loss: 7.5949 - accuracy: 0.5047
 8800/25000 [=========>....................] - ETA: 1:00 - loss: 7.5952 - accuracy: 0.5047
 8832/25000 [=========>....................] - ETA: 1:00 - loss: 7.5937 - accuracy: 0.5048
 8864/25000 [=========>....................] - ETA: 59s - loss: 7.5922 - accuracy: 0.5049 
 8896/25000 [=========>....................] - ETA: 59s - loss: 7.5977 - accuracy: 0.5045
 8928/25000 [=========>....................] - ETA: 59s - loss: 7.5996 - accuracy: 0.5044
 8960/25000 [=========>....................] - ETA: 59s - loss: 7.5947 - accuracy: 0.5047
 8992/25000 [=========>....................] - ETA: 59s - loss: 7.5933 - accuracy: 0.5048
 9024/25000 [=========>....................] - ETA: 59s - loss: 7.5953 - accuracy: 0.5047
 9056/25000 [=========>....................] - ETA: 59s - loss: 7.5955 - accuracy: 0.5046
 9088/25000 [=========>....................] - ETA: 59s - loss: 7.5974 - accuracy: 0.5045
 9120/25000 [=========>....................] - ETA: 59s - loss: 7.5960 - accuracy: 0.5046
 9152/25000 [=========>....................] - ETA: 58s - loss: 7.5946 - accuracy: 0.5047
 9184/25000 [==========>...................] - ETA: 58s - loss: 7.5948 - accuracy: 0.5047
 9216/25000 [==========>...................] - ETA: 58s - loss: 7.6017 - accuracy: 0.5042
 9248/25000 [==========>...................] - ETA: 58s - loss: 7.6020 - accuracy: 0.5042
 9280/25000 [==========>...................] - ETA: 58s - loss: 7.5972 - accuracy: 0.5045
 9312/25000 [==========>...................] - ETA: 58s - loss: 7.5991 - accuracy: 0.5044
 9344/25000 [==========>...................] - ETA: 58s - loss: 7.6059 - accuracy: 0.5040
 9376/25000 [==========>...................] - ETA: 58s - loss: 7.6159 - accuracy: 0.5033
 9408/25000 [==========>...................] - ETA: 57s - loss: 7.6226 - accuracy: 0.5029
 9440/25000 [==========>...................] - ETA: 57s - loss: 7.6244 - accuracy: 0.5028
 9472/25000 [==========>...................] - ETA: 57s - loss: 7.6278 - accuracy: 0.5025
 9504/25000 [==========>...................] - ETA: 57s - loss: 7.6166 - accuracy: 0.5033
 9536/25000 [==========>...................] - ETA: 57s - loss: 7.6200 - accuracy: 0.5030
 9568/25000 [==========>...................] - ETA: 57s - loss: 7.6137 - accuracy: 0.5034
 9600/25000 [==========>...................] - ETA: 57s - loss: 7.6155 - accuracy: 0.5033
 9632/25000 [==========>...................] - ETA: 57s - loss: 7.6189 - accuracy: 0.5031
 9664/25000 [==========>...................] - ETA: 57s - loss: 7.6222 - accuracy: 0.5029
 9696/25000 [==========>...................] - ETA: 56s - loss: 7.6287 - accuracy: 0.5025
 9728/25000 [==========>...................] - ETA: 56s - loss: 7.6288 - accuracy: 0.5025
 9760/25000 [==========>...................] - ETA: 56s - loss: 7.6305 - accuracy: 0.5024
 9792/25000 [==========>...................] - ETA: 56s - loss: 7.6306 - accuracy: 0.5023
 9824/25000 [==========>...................] - ETA: 56s - loss: 7.6354 - accuracy: 0.5020
 9856/25000 [==========>...................] - ETA: 56s - loss: 7.6355 - accuracy: 0.5020
 9888/25000 [==========>...................] - ETA: 56s - loss: 7.6356 - accuracy: 0.5020
 9920/25000 [==========>...................] - ETA: 56s - loss: 7.6388 - accuracy: 0.5018
 9952/25000 [==========>...................] - ETA: 56s - loss: 7.6404 - accuracy: 0.5017
 9984/25000 [==========>...................] - ETA: 55s - loss: 7.6390 - accuracy: 0.5018
10016/25000 [===========>..................] - ETA: 55s - loss: 7.6360 - accuracy: 0.5020
10048/25000 [===========>..................] - ETA: 55s - loss: 7.6422 - accuracy: 0.5016
10080/25000 [===========>..................] - ETA: 55s - loss: 7.6499 - accuracy: 0.5011
10112/25000 [===========>..................] - ETA: 55s - loss: 7.6530 - accuracy: 0.5009
10144/25000 [===========>..................] - ETA: 55s - loss: 7.6576 - accuracy: 0.5006
10176/25000 [===========>..................] - ETA: 55s - loss: 7.6606 - accuracy: 0.5004
10208/25000 [===========>..................] - ETA: 55s - loss: 7.6636 - accuracy: 0.5002
10240/25000 [===========>..................] - ETA: 54s - loss: 7.6711 - accuracy: 0.4997
10272/25000 [===========>..................] - ETA: 54s - loss: 7.6726 - accuracy: 0.4996
10304/25000 [===========>..................] - ETA: 54s - loss: 7.6696 - accuracy: 0.4998
10336/25000 [===========>..................] - ETA: 54s - loss: 7.6755 - accuracy: 0.4994
10368/25000 [===========>..................] - ETA: 54s - loss: 7.6681 - accuracy: 0.4999
10400/25000 [===========>..................] - ETA: 54s - loss: 7.6755 - accuracy: 0.4994
10432/25000 [===========>..................] - ETA: 54s - loss: 7.6696 - accuracy: 0.4998
10464/25000 [===========>..................] - ETA: 54s - loss: 7.6681 - accuracy: 0.4999
10496/25000 [===========>..................] - ETA: 53s - loss: 7.6725 - accuracy: 0.4996
10528/25000 [===========>..................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
10560/25000 [===========>..................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
10592/25000 [===========>..................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
10624/25000 [===========>..................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
10656/25000 [===========>..................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
10688/25000 [===========>..................] - ETA: 53s - loss: 7.6652 - accuracy: 0.5001
10720/25000 [===========>..................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
10752/25000 [===========>..................] - ETA: 53s - loss: 7.6638 - accuracy: 0.5002
10784/25000 [===========>..................] - ETA: 52s - loss: 7.6709 - accuracy: 0.4997
10816/25000 [===========>..................] - ETA: 52s - loss: 7.6624 - accuracy: 0.5003
10848/25000 [============>.................] - ETA: 52s - loss: 7.6567 - accuracy: 0.5006
10880/25000 [============>.................] - ETA: 52s - loss: 7.6511 - accuracy: 0.5010
10912/25000 [============>.................] - ETA: 52s - loss: 7.6469 - accuracy: 0.5013
10944/25000 [============>.................] - ETA: 52s - loss: 7.6484 - accuracy: 0.5012
10976/25000 [============>.................] - ETA: 52s - loss: 7.6471 - accuracy: 0.5013
11008/25000 [============>.................] - ETA: 52s - loss: 7.6457 - accuracy: 0.5014
11040/25000 [============>.................] - ETA: 51s - loss: 7.6444 - accuracy: 0.5014
11072/25000 [============>.................] - ETA: 51s - loss: 7.6472 - accuracy: 0.5013
11104/25000 [============>.................] - ETA: 51s - loss: 7.6431 - accuracy: 0.5015
11136/25000 [============>.................] - ETA: 51s - loss: 7.6391 - accuracy: 0.5018
11168/25000 [============>.................] - ETA: 51s - loss: 7.6378 - accuracy: 0.5019
11200/25000 [============>.................] - ETA: 51s - loss: 7.6447 - accuracy: 0.5014
11232/25000 [============>.................] - ETA: 51s - loss: 7.6407 - accuracy: 0.5017
11264/25000 [============>.................] - ETA: 51s - loss: 7.6435 - accuracy: 0.5015
11296/25000 [============>.................] - ETA: 51s - loss: 7.6463 - accuracy: 0.5013
11328/25000 [============>.................] - ETA: 50s - loss: 7.6409 - accuracy: 0.5017
11360/25000 [============>.................] - ETA: 50s - loss: 7.6410 - accuracy: 0.5017
11392/25000 [============>.................] - ETA: 50s - loss: 7.6451 - accuracy: 0.5014
11424/25000 [============>.................] - ETA: 50s - loss: 7.6425 - accuracy: 0.5016
11456/25000 [============>.................] - ETA: 50s - loss: 7.6439 - accuracy: 0.5015
11488/25000 [============>.................] - ETA: 50s - loss: 7.6399 - accuracy: 0.5017
11520/25000 [============>.................] - ETA: 50s - loss: 7.6440 - accuracy: 0.5015
11552/25000 [============>.................] - ETA: 50s - loss: 7.6507 - accuracy: 0.5010
11584/25000 [============>.................] - ETA: 49s - loss: 7.6468 - accuracy: 0.5013
11616/25000 [============>.................] - ETA: 49s - loss: 7.6468 - accuracy: 0.5013
11648/25000 [============>.................] - ETA: 49s - loss: 7.6508 - accuracy: 0.5010
11680/25000 [=============>................] - ETA: 49s - loss: 7.6482 - accuracy: 0.5012
11712/25000 [=============>................] - ETA: 49s - loss: 7.6483 - accuracy: 0.5012
11744/25000 [=============>................] - ETA: 49s - loss: 7.6483 - accuracy: 0.5012
11776/25000 [=============>................] - ETA: 49s - loss: 7.6471 - accuracy: 0.5013
11808/25000 [=============>................] - ETA: 49s - loss: 7.6471 - accuracy: 0.5013
11840/25000 [=============>................] - ETA: 48s - loss: 7.6472 - accuracy: 0.5013
11872/25000 [=============>................] - ETA: 48s - loss: 7.6485 - accuracy: 0.5012
11904/25000 [=============>................] - ETA: 48s - loss: 7.6473 - accuracy: 0.5013
11936/25000 [=============>................] - ETA: 48s - loss: 7.6512 - accuracy: 0.5010
11968/25000 [=============>................] - ETA: 48s - loss: 7.6474 - accuracy: 0.5013
12000/25000 [=============>................] - ETA: 48s - loss: 7.6487 - accuracy: 0.5012
12032/25000 [=============>................] - ETA: 48s - loss: 7.6513 - accuracy: 0.5010
12064/25000 [=============>................] - ETA: 48s - loss: 7.6526 - accuracy: 0.5009
12096/25000 [=============>................] - ETA: 47s - loss: 7.6527 - accuracy: 0.5009
12128/25000 [=============>................] - ETA: 47s - loss: 7.6565 - accuracy: 0.5007
12160/25000 [=============>................] - ETA: 47s - loss: 7.6553 - accuracy: 0.5007
12192/25000 [=============>................] - ETA: 47s - loss: 7.6578 - accuracy: 0.5006
12224/25000 [=============>................] - ETA: 47s - loss: 7.6503 - accuracy: 0.5011
12256/25000 [=============>................] - ETA: 47s - loss: 7.6554 - accuracy: 0.5007
12288/25000 [=============>................] - ETA: 47s - loss: 7.6579 - accuracy: 0.5006
12320/25000 [=============>................] - ETA: 47s - loss: 7.6592 - accuracy: 0.5005
12352/25000 [=============>................] - ETA: 47s - loss: 7.6617 - accuracy: 0.5003
12384/25000 [=============>................] - ETA: 46s - loss: 7.6679 - accuracy: 0.4999
12416/25000 [=============>................] - ETA: 46s - loss: 7.6691 - accuracy: 0.4998
12448/25000 [=============>................] - ETA: 46s - loss: 7.6728 - accuracy: 0.4996
12480/25000 [=============>................] - ETA: 46s - loss: 7.6728 - accuracy: 0.4996
12512/25000 [==============>...............] - ETA: 46s - loss: 7.6764 - accuracy: 0.4994
12544/25000 [==============>...............] - ETA: 46s - loss: 7.6703 - accuracy: 0.4998
12576/25000 [==============>...............] - ETA: 46s - loss: 7.6666 - accuracy: 0.5000
12608/25000 [==============>...............] - ETA: 46s - loss: 7.6630 - accuracy: 0.5002
12640/25000 [==============>...............] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
12672/25000 [==============>...............] - ETA: 45s - loss: 7.6654 - accuracy: 0.5001
12704/25000 [==============>...............] - ETA: 45s - loss: 7.6702 - accuracy: 0.4998
12736/25000 [==============>...............] - ETA: 45s - loss: 7.6690 - accuracy: 0.4998
12768/25000 [==============>...............] - ETA: 45s - loss: 7.6726 - accuracy: 0.4996
12800/25000 [==============>...............] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
12832/25000 [==============>...............] - ETA: 45s - loss: 7.6678 - accuracy: 0.4999
12864/25000 [==============>...............] - ETA: 45s - loss: 7.6726 - accuracy: 0.4996
12896/25000 [==============>...............] - ETA: 44s - loss: 7.6726 - accuracy: 0.4996
12928/25000 [==============>...............] - ETA: 44s - loss: 7.6714 - accuracy: 0.4997
12960/25000 [==============>...............] - ETA: 44s - loss: 7.6678 - accuracy: 0.4999
12992/25000 [==============>...............] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
13024/25000 [==============>...............] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
13056/25000 [==============>...............] - ETA: 44s - loss: 7.6737 - accuracy: 0.4995
13088/25000 [==============>...............] - ETA: 44s - loss: 7.6725 - accuracy: 0.4996
13120/25000 [==============>...............] - ETA: 44s - loss: 7.6701 - accuracy: 0.4998
13152/25000 [==============>...............] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
13184/25000 [==============>...............] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
13216/25000 [==============>...............] - ETA: 43s - loss: 7.6643 - accuracy: 0.5002
13248/25000 [==============>...............] - ETA: 43s - loss: 7.6724 - accuracy: 0.4996
13280/25000 [==============>...............] - ETA: 43s - loss: 7.6689 - accuracy: 0.4998
13312/25000 [==============>...............] - ETA: 43s - loss: 7.6689 - accuracy: 0.4998
13344/25000 [===============>..............] - ETA: 43s - loss: 7.6689 - accuracy: 0.4999
13376/25000 [===============>..............] - ETA: 43s - loss: 7.6678 - accuracy: 0.4999
13408/25000 [===============>..............] - ETA: 43s - loss: 7.6643 - accuracy: 0.5001
13440/25000 [===============>..............] - ETA: 42s - loss: 7.6643 - accuracy: 0.5001
13472/25000 [===============>..............] - ETA: 42s - loss: 7.6621 - accuracy: 0.5003
13504/25000 [===============>..............] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
13536/25000 [===============>..............] - ETA: 42s - loss: 7.6644 - accuracy: 0.5001
13568/25000 [===============>..............] - ETA: 42s - loss: 7.6610 - accuracy: 0.5004
13600/25000 [===============>..............] - ETA: 42s - loss: 7.6621 - accuracy: 0.5003
13632/25000 [===============>..............] - ETA: 42s - loss: 7.6599 - accuracy: 0.5004
13664/25000 [===============>..............] - ETA: 42s - loss: 7.6588 - accuracy: 0.5005
13696/25000 [===============>..............] - ETA: 42s - loss: 7.6599 - accuracy: 0.5004
13728/25000 [===============>..............] - ETA: 41s - loss: 7.6543 - accuracy: 0.5008
13760/25000 [===============>..............] - ETA: 41s - loss: 7.6521 - accuracy: 0.5009
13792/25000 [===============>..............] - ETA: 41s - loss: 7.6511 - accuracy: 0.5010
13824/25000 [===============>..............] - ETA: 41s - loss: 7.6544 - accuracy: 0.5008
13856/25000 [===============>..............] - ETA: 41s - loss: 7.6544 - accuracy: 0.5008
13888/25000 [===============>..............] - ETA: 41s - loss: 7.6467 - accuracy: 0.5013
13920/25000 [===============>..............] - ETA: 41s - loss: 7.6490 - accuracy: 0.5011
13952/25000 [===============>..............] - ETA: 41s - loss: 7.6479 - accuracy: 0.5012
13984/25000 [===============>..............] - ETA: 40s - loss: 7.6458 - accuracy: 0.5014
14016/25000 [===============>..............] - ETA: 40s - loss: 7.6480 - accuracy: 0.5012
14048/25000 [===============>..............] - ETA: 40s - loss: 7.6481 - accuracy: 0.5012
14080/25000 [===============>..............] - ETA: 40s - loss: 7.6481 - accuracy: 0.5012
14112/25000 [===============>..............] - ETA: 40s - loss: 7.6514 - accuracy: 0.5010
14144/25000 [===============>..............] - ETA: 40s - loss: 7.6547 - accuracy: 0.5008
14176/25000 [================>.............] - ETA: 40s - loss: 7.6558 - accuracy: 0.5007
14208/25000 [================>.............] - ETA: 40s - loss: 7.6526 - accuracy: 0.5009
14240/25000 [================>.............] - ETA: 39s - loss: 7.6526 - accuracy: 0.5009
14272/25000 [================>.............] - ETA: 39s - loss: 7.6527 - accuracy: 0.5009
14304/25000 [================>.............] - ETA: 39s - loss: 7.6548 - accuracy: 0.5008
14336/25000 [================>.............] - ETA: 39s - loss: 7.6495 - accuracy: 0.5011
14368/25000 [================>.............] - ETA: 39s - loss: 7.6549 - accuracy: 0.5008
14400/25000 [================>.............] - ETA: 39s - loss: 7.6528 - accuracy: 0.5009
14432/25000 [================>.............] - ETA: 39s - loss: 7.6539 - accuracy: 0.5008
14464/25000 [================>.............] - ETA: 39s - loss: 7.6560 - accuracy: 0.5007
14496/25000 [================>.............] - ETA: 39s - loss: 7.6634 - accuracy: 0.5002
14528/25000 [================>.............] - ETA: 38s - loss: 7.6666 - accuracy: 0.5000
14560/25000 [================>.............] - ETA: 38s - loss: 7.6656 - accuracy: 0.5001
14592/25000 [================>.............] - ETA: 38s - loss: 7.6677 - accuracy: 0.4999
14624/25000 [================>.............] - ETA: 38s - loss: 7.6719 - accuracy: 0.4997
14656/25000 [================>.............] - ETA: 38s - loss: 7.6687 - accuracy: 0.4999
14688/25000 [================>.............] - ETA: 38s - loss: 7.6656 - accuracy: 0.5001
14720/25000 [================>.............] - ETA: 38s - loss: 7.6645 - accuracy: 0.5001
14752/25000 [================>.............] - ETA: 38s - loss: 7.6604 - accuracy: 0.5004
14784/25000 [================>.............] - ETA: 37s - loss: 7.6625 - accuracy: 0.5003
14816/25000 [================>.............] - ETA: 37s - loss: 7.6614 - accuracy: 0.5003
14848/25000 [================>.............] - ETA: 37s - loss: 7.6635 - accuracy: 0.5002
14880/25000 [================>.............] - ETA: 37s - loss: 7.6635 - accuracy: 0.5002
14912/25000 [================>.............] - ETA: 37s - loss: 7.6635 - accuracy: 0.5002
14944/25000 [================>.............] - ETA: 37s - loss: 7.6594 - accuracy: 0.5005
14976/25000 [================>.............] - ETA: 37s - loss: 7.6595 - accuracy: 0.5005
15008/25000 [=================>............] - ETA: 37s - loss: 7.6584 - accuracy: 0.5005
15040/25000 [=================>............] - ETA: 37s - loss: 7.6574 - accuracy: 0.5006
15072/25000 [=================>............] - ETA: 36s - loss: 7.6575 - accuracy: 0.5006
15104/25000 [=================>............] - ETA: 36s - loss: 7.6544 - accuracy: 0.5008
15136/25000 [=================>............] - ETA: 36s - loss: 7.6524 - accuracy: 0.5009
15168/25000 [=================>............] - ETA: 36s - loss: 7.6494 - accuracy: 0.5011
15200/25000 [=================>............] - ETA: 36s - loss: 7.6505 - accuracy: 0.5011
15232/25000 [=================>............] - ETA: 36s - loss: 7.6475 - accuracy: 0.5012
15264/25000 [=================>............] - ETA: 36s - loss: 7.6485 - accuracy: 0.5012
15296/25000 [=================>............] - ETA: 36s - loss: 7.6466 - accuracy: 0.5013
15328/25000 [=================>............] - ETA: 35s - loss: 7.6486 - accuracy: 0.5012
15360/25000 [=================>............] - ETA: 35s - loss: 7.6477 - accuracy: 0.5012
15392/25000 [=================>............] - ETA: 35s - loss: 7.6497 - accuracy: 0.5011
15424/25000 [=================>............] - ETA: 35s - loss: 7.6477 - accuracy: 0.5012
15456/25000 [=================>............] - ETA: 35s - loss: 7.6458 - accuracy: 0.5014
15488/25000 [=================>............] - ETA: 35s - loss: 7.6438 - accuracy: 0.5015
15520/25000 [=================>............] - ETA: 35s - loss: 7.6419 - accuracy: 0.5016
15552/25000 [=================>............] - ETA: 35s - loss: 7.6449 - accuracy: 0.5014
15584/25000 [=================>............] - ETA: 35s - loss: 7.6460 - accuracy: 0.5013
15616/25000 [=================>............] - ETA: 34s - loss: 7.6470 - accuracy: 0.5013
15648/25000 [=================>............] - ETA: 34s - loss: 7.6382 - accuracy: 0.5019
15680/25000 [=================>............] - ETA: 34s - loss: 7.6441 - accuracy: 0.5015
15712/25000 [=================>............] - ETA: 34s - loss: 7.6432 - accuracy: 0.5015
15744/25000 [=================>............] - ETA: 34s - loss: 7.6442 - accuracy: 0.5015
15776/25000 [=================>............] - ETA: 34s - loss: 7.6443 - accuracy: 0.5015
15808/25000 [=================>............] - ETA: 34s - loss: 7.6414 - accuracy: 0.5016
15840/25000 [==================>...........] - ETA: 34s - loss: 7.6415 - accuracy: 0.5016
15872/25000 [==================>...........] - ETA: 33s - loss: 7.6415 - accuracy: 0.5016
15904/25000 [==================>...........] - ETA: 33s - loss: 7.6425 - accuracy: 0.5016
15936/25000 [==================>...........] - ETA: 33s - loss: 7.6435 - accuracy: 0.5015
15968/25000 [==================>...........] - ETA: 33s - loss: 7.6426 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 33s - loss: 7.6427 - accuracy: 0.5016
16032/25000 [==================>...........] - ETA: 33s - loss: 7.6456 - accuracy: 0.5014
16064/25000 [==================>...........] - ETA: 33s - loss: 7.6494 - accuracy: 0.5011
16096/25000 [==================>...........] - ETA: 33s - loss: 7.6447 - accuracy: 0.5014
16128/25000 [==================>...........] - ETA: 33s - loss: 7.6457 - accuracy: 0.5014
16160/25000 [==================>...........] - ETA: 32s - loss: 7.6448 - accuracy: 0.5014
16192/25000 [==================>...........] - ETA: 32s - loss: 7.6439 - accuracy: 0.5015
16224/25000 [==================>...........] - ETA: 32s - loss: 7.6402 - accuracy: 0.5017
16256/25000 [==================>...........] - ETA: 32s - loss: 7.6374 - accuracy: 0.5019
16288/25000 [==================>...........] - ETA: 32s - loss: 7.6403 - accuracy: 0.5017
16320/25000 [==================>...........] - ETA: 32s - loss: 7.6431 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 32s - loss: 7.6441 - accuracy: 0.5015
16384/25000 [==================>...........] - ETA: 32s - loss: 7.6404 - accuracy: 0.5017
16416/25000 [==================>...........] - ETA: 31s - loss: 7.6386 - accuracy: 0.5018
16448/25000 [==================>...........] - ETA: 31s - loss: 7.6396 - accuracy: 0.5018
16480/25000 [==================>...........] - ETA: 31s - loss: 7.6378 - accuracy: 0.5019
16512/25000 [==================>...........] - ETA: 31s - loss: 7.6397 - accuracy: 0.5018
16544/25000 [==================>...........] - ETA: 31s - loss: 7.6416 - accuracy: 0.5016
16576/25000 [==================>...........] - ETA: 31s - loss: 7.6398 - accuracy: 0.5017
16608/25000 [==================>...........] - ETA: 31s - loss: 7.6417 - accuracy: 0.5016
16640/25000 [==================>...........] - ETA: 31s - loss: 7.6408 - accuracy: 0.5017
16672/25000 [===================>..........] - ETA: 31s - loss: 7.6381 - accuracy: 0.5019
16704/25000 [===================>..........] - ETA: 30s - loss: 7.6345 - accuracy: 0.5021
16736/25000 [===================>..........] - ETA: 30s - loss: 7.6318 - accuracy: 0.5023
16768/25000 [===================>..........] - ETA: 30s - loss: 7.6328 - accuracy: 0.5022
16800/25000 [===================>..........] - ETA: 30s - loss: 7.6338 - accuracy: 0.5021
16832/25000 [===================>..........] - ETA: 30s - loss: 7.6329 - accuracy: 0.5022
16864/25000 [===================>..........] - ETA: 30s - loss: 7.6339 - accuracy: 0.5021
16896/25000 [===================>..........] - ETA: 30s - loss: 7.6358 - accuracy: 0.5020
16928/25000 [===================>..........] - ETA: 30s - loss: 7.6413 - accuracy: 0.5017
16960/25000 [===================>..........] - ETA: 29s - loss: 7.6413 - accuracy: 0.5017
16992/25000 [===================>..........] - ETA: 29s - loss: 7.6414 - accuracy: 0.5016
17024/25000 [===================>..........] - ETA: 29s - loss: 7.6387 - accuracy: 0.5018
17056/25000 [===================>..........] - ETA: 29s - loss: 7.6388 - accuracy: 0.5018
17088/25000 [===================>..........] - ETA: 29s - loss: 7.6406 - accuracy: 0.5017
17120/25000 [===================>..........] - ETA: 29s - loss: 7.6371 - accuracy: 0.5019
17152/25000 [===================>..........] - ETA: 29s - loss: 7.6380 - accuracy: 0.5019
17184/25000 [===================>..........] - ETA: 29s - loss: 7.6372 - accuracy: 0.5019
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6363 - accuracy: 0.5020
17248/25000 [===================>..........] - ETA: 28s - loss: 7.6355 - accuracy: 0.5020
17280/25000 [===================>..........] - ETA: 28s - loss: 7.6364 - accuracy: 0.5020
17312/25000 [===================>..........] - ETA: 28s - loss: 7.6392 - accuracy: 0.5018
17344/25000 [===================>..........] - ETA: 28s - loss: 7.6401 - accuracy: 0.5017
17376/25000 [===================>..........] - ETA: 28s - loss: 7.6419 - accuracy: 0.5016
17408/25000 [===================>..........] - ETA: 28s - loss: 7.6420 - accuracy: 0.5016
17440/25000 [===================>..........] - ETA: 28s - loss: 7.6420 - accuracy: 0.5016
17472/25000 [===================>..........] - ETA: 28s - loss: 7.6429 - accuracy: 0.5015
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6412 - accuracy: 0.5017
17536/25000 [====================>.........] - ETA: 27s - loss: 7.6421 - accuracy: 0.5016
17568/25000 [====================>.........] - ETA: 27s - loss: 7.6422 - accuracy: 0.5016
17600/25000 [====================>.........] - ETA: 27s - loss: 7.6379 - accuracy: 0.5019
17632/25000 [====================>.........] - ETA: 27s - loss: 7.6362 - accuracy: 0.5020
17664/25000 [====================>.........] - ETA: 27s - loss: 7.6336 - accuracy: 0.5022
17696/25000 [====================>.........] - ETA: 27s - loss: 7.6320 - accuracy: 0.5023
17728/25000 [====================>.........] - ETA: 27s - loss: 7.6372 - accuracy: 0.5019
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6399 - accuracy: 0.5017
17792/25000 [====================>.........] - ETA: 26s - loss: 7.6399 - accuracy: 0.5017
17824/25000 [====================>.........] - ETA: 26s - loss: 7.6374 - accuracy: 0.5019
17856/25000 [====================>.........] - ETA: 26s - loss: 7.6357 - accuracy: 0.5020
17888/25000 [====================>.........] - ETA: 26s - loss: 7.6358 - accuracy: 0.5020
17920/25000 [====================>.........] - ETA: 26s - loss: 7.6392 - accuracy: 0.5018
17952/25000 [====================>.........] - ETA: 26s - loss: 7.6350 - accuracy: 0.5021
17984/25000 [====================>.........] - ETA: 26s - loss: 7.6376 - accuracy: 0.5019
18016/25000 [====================>.........] - ETA: 26s - loss: 7.6394 - accuracy: 0.5018
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6411 - accuracy: 0.5017
18080/25000 [====================>.........] - ETA: 25s - loss: 7.6437 - accuracy: 0.5015
18112/25000 [====================>.........] - ETA: 25s - loss: 7.6480 - accuracy: 0.5012
18144/25000 [====================>.........] - ETA: 25s - loss: 7.6497 - accuracy: 0.5011
18176/25000 [====================>.........] - ETA: 25s - loss: 7.6506 - accuracy: 0.5010
18208/25000 [====================>.........] - ETA: 25s - loss: 7.6506 - accuracy: 0.5010
18240/25000 [====================>.........] - ETA: 25s - loss: 7.6498 - accuracy: 0.5011
18272/25000 [====================>.........] - ETA: 25s - loss: 7.6498 - accuracy: 0.5011
18304/25000 [====================>.........] - ETA: 24s - loss: 7.6482 - accuracy: 0.5012
18336/25000 [=====================>........] - ETA: 24s - loss: 7.6457 - accuracy: 0.5014
18368/25000 [=====================>........] - ETA: 24s - loss: 7.6432 - accuracy: 0.5015
18400/25000 [=====================>........] - ETA: 24s - loss: 7.6433 - accuracy: 0.5015
18432/25000 [=====================>........] - ETA: 24s - loss: 7.6491 - accuracy: 0.5011
18464/25000 [=====================>........] - ETA: 24s - loss: 7.6483 - accuracy: 0.5012
18496/25000 [=====================>........] - ETA: 24s - loss: 7.6442 - accuracy: 0.5015
18528/25000 [=====================>........] - ETA: 24s - loss: 7.6476 - accuracy: 0.5012
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6493 - accuracy: 0.5011
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6468 - accuracy: 0.5013
18624/25000 [=====================>........] - ETA: 23s - loss: 7.6452 - accuracy: 0.5014
18656/25000 [=====================>........] - ETA: 23s - loss: 7.6444 - accuracy: 0.5014
18688/25000 [=====================>........] - ETA: 23s - loss: 7.6420 - accuracy: 0.5016
18720/25000 [=====================>........] - ETA: 23s - loss: 7.6420 - accuracy: 0.5016
18752/25000 [=====================>........] - ETA: 23s - loss: 7.6405 - accuracy: 0.5017
18784/25000 [=====================>........] - ETA: 23s - loss: 7.6446 - accuracy: 0.5014
18816/25000 [=====================>........] - ETA: 23s - loss: 7.6479 - accuracy: 0.5012
18848/25000 [=====================>........] - ETA: 22s - loss: 7.6471 - accuracy: 0.5013
18880/25000 [=====================>........] - ETA: 22s - loss: 7.6431 - accuracy: 0.5015
18912/25000 [=====================>........] - ETA: 22s - loss: 7.6464 - accuracy: 0.5013
18944/25000 [=====================>........] - ETA: 22s - loss: 7.6431 - accuracy: 0.5015
18976/25000 [=====================>........] - ETA: 22s - loss: 7.6416 - accuracy: 0.5016
19008/25000 [=====================>........] - ETA: 22s - loss: 7.6416 - accuracy: 0.5016
19040/25000 [=====================>........] - ETA: 22s - loss: 7.6408 - accuracy: 0.5017
19072/25000 [=====================>........] - ETA: 22s - loss: 7.6393 - accuracy: 0.5018
19104/25000 [=====================>........] - ETA: 21s - loss: 7.6425 - accuracy: 0.5016
19136/25000 [=====================>........] - ETA: 21s - loss: 7.6394 - accuracy: 0.5018
19168/25000 [======================>.......] - ETA: 21s - loss: 7.6402 - accuracy: 0.5017
19200/25000 [======================>.......] - ETA: 21s - loss: 7.6379 - accuracy: 0.5019
19232/25000 [======================>.......] - ETA: 21s - loss: 7.6371 - accuracy: 0.5019
19264/25000 [======================>.......] - ETA: 21s - loss: 7.6388 - accuracy: 0.5018
19296/25000 [======================>.......] - ETA: 21s - loss: 7.6348 - accuracy: 0.5021
19328/25000 [======================>.......] - ETA: 21s - loss: 7.6373 - accuracy: 0.5019
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6397 - accuracy: 0.5018
19392/25000 [======================>.......] - ETA: 20s - loss: 7.6382 - accuracy: 0.5019
19424/25000 [======================>.......] - ETA: 20s - loss: 7.6398 - accuracy: 0.5018
19456/25000 [======================>.......] - ETA: 20s - loss: 7.6438 - accuracy: 0.5015
19488/25000 [======================>.......] - ETA: 20s - loss: 7.6430 - accuracy: 0.5015
19520/25000 [======================>.......] - ETA: 20s - loss: 7.6399 - accuracy: 0.5017
19552/25000 [======================>.......] - ETA: 20s - loss: 7.6407 - accuracy: 0.5017
19584/25000 [======================>.......] - ETA: 20s - loss: 7.6408 - accuracy: 0.5017
19616/25000 [======================>.......] - ETA: 20s - loss: 7.6416 - accuracy: 0.5016
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6448 - accuracy: 0.5014
19680/25000 [======================>.......] - ETA: 19s - loss: 7.6448 - accuracy: 0.5014
19712/25000 [======================>.......] - ETA: 19s - loss: 7.6456 - accuracy: 0.5014
19744/25000 [======================>.......] - ETA: 19s - loss: 7.6464 - accuracy: 0.5013
19776/25000 [======================>.......] - ETA: 19s - loss: 7.6488 - accuracy: 0.5012
19808/25000 [======================>.......] - ETA: 19s - loss: 7.6496 - accuracy: 0.5011
19840/25000 [======================>.......] - ETA: 19s - loss: 7.6488 - accuracy: 0.5012
19872/25000 [======================>.......] - ETA: 19s - loss: 7.6481 - accuracy: 0.5012
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6504 - accuracy: 0.5011
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6520 - accuracy: 0.5010
19968/25000 [======================>.......] - ETA: 18s - loss: 7.6505 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 18s - loss: 7.6528 - accuracy: 0.5009
20032/25000 [=======================>......] - ETA: 18s - loss: 7.6505 - accuracy: 0.5010
20064/25000 [=======================>......] - ETA: 18s - loss: 7.6536 - accuracy: 0.5008
20096/25000 [=======================>......] - ETA: 18s - loss: 7.6536 - accuracy: 0.5008
20128/25000 [=======================>......] - ETA: 18s - loss: 7.6544 - accuracy: 0.5008
20160/25000 [=======================>......] - ETA: 18s - loss: 7.6537 - accuracy: 0.5008
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6537 - accuracy: 0.5008
20224/25000 [=======================>......] - ETA: 17s - loss: 7.6545 - accuracy: 0.5008
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6560 - accuracy: 0.5007
20288/25000 [=======================>......] - ETA: 17s - loss: 7.6576 - accuracy: 0.5006
20320/25000 [=======================>......] - ETA: 17s - loss: 7.6583 - accuracy: 0.5005
20352/25000 [=======================>......] - ETA: 17s - loss: 7.6561 - accuracy: 0.5007
20384/25000 [=======================>......] - ETA: 17s - loss: 7.6561 - accuracy: 0.5007
20416/25000 [=======================>......] - ETA: 17s - loss: 7.6561 - accuracy: 0.5007
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6614 - accuracy: 0.5003
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
20512/25000 [=======================>......] - ETA: 16s - loss: 7.6644 - accuracy: 0.5001
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6614 - accuracy: 0.5003
20576/25000 [=======================>......] - ETA: 16s - loss: 7.6644 - accuracy: 0.5001
20608/25000 [=======================>......] - ETA: 16s - loss: 7.6674 - accuracy: 0.5000
20640/25000 [=======================>......] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
20672/25000 [=======================>......] - ETA: 16s - loss: 7.6696 - accuracy: 0.4998
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6718 - accuracy: 0.4997
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6711 - accuracy: 0.4997
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6718 - accuracy: 0.4997
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6710 - accuracy: 0.4997
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6696 - accuracy: 0.4998
20864/25000 [========================>.....] - ETA: 15s - loss: 7.6696 - accuracy: 0.4998
20896/25000 [========================>.....] - ETA: 15s - loss: 7.6710 - accuracy: 0.4997
20928/25000 [========================>.....] - ETA: 15s - loss: 7.6681 - accuracy: 0.4999
20960/25000 [========================>.....] - ETA: 15s - loss: 7.6681 - accuracy: 0.4999
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6710 - accuracy: 0.4997
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6725 - accuracy: 0.4996
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6688 - accuracy: 0.4999
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6710 - accuracy: 0.4997
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6746 - accuracy: 0.4995
21152/25000 [========================>.....] - ETA: 14s - loss: 7.6717 - accuracy: 0.4997
21184/25000 [========================>.....] - ETA: 14s - loss: 7.6760 - accuracy: 0.4994
21216/25000 [========================>.....] - ETA: 14s - loss: 7.6789 - accuracy: 0.4992
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6774 - accuracy: 0.4993
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6789 - accuracy: 0.4992
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6832 - accuracy: 0.4989
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6810 - accuracy: 0.4991
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6817 - accuracy: 0.4990
21408/25000 [========================>.....] - ETA: 13s - loss: 7.6788 - accuracy: 0.4992
21440/25000 [========================>.....] - ETA: 13s - loss: 7.6809 - accuracy: 0.4991
21472/25000 [========================>.....] - ETA: 13s - loss: 7.6795 - accuracy: 0.4992
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6780 - accuracy: 0.4993
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6801 - accuracy: 0.4991
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6801 - accuracy: 0.4991
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6794 - accuracy: 0.4992
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6787 - accuracy: 0.4992
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6808 - accuracy: 0.4991
21696/25000 [=========================>....] - ETA: 12s - loss: 7.6864 - accuracy: 0.4987
21728/25000 [=========================>....] - ETA: 12s - loss: 7.6836 - accuracy: 0.4989
21760/25000 [=========================>....] - ETA: 12s - loss: 7.6849 - accuracy: 0.4988
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6856 - accuracy: 0.4988
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6870 - accuracy: 0.4987
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6863 - accuracy: 0.4987
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6841 - accuracy: 0.4989
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6876 - accuracy: 0.4986
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6862 - accuracy: 0.4987
21984/25000 [=========================>....] - ETA: 11s - loss: 7.6848 - accuracy: 0.4988
22016/25000 [=========================>....] - ETA: 11s - loss: 7.6833 - accuracy: 0.4989
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6805 - accuracy: 0.4991
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6763 - accuracy: 0.4994
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6742 - accuracy: 0.4995
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6701 - accuracy: 0.4998
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6728 - accuracy: 0.4996
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6715 - accuracy: 0.4997
22240/25000 [=========================>....] - ETA: 10s - loss: 7.6742 - accuracy: 0.4995
22272/25000 [=========================>....] - ETA: 10s - loss: 7.6756 - accuracy: 0.4994
22304/25000 [=========================>....] - ETA: 10s - loss: 7.6742 - accuracy: 0.4995
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6783 - accuracy: 0.4992 
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6776 - accuracy: 0.4993
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6741 - accuracy: 0.4995
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6728 - accuracy: 0.4996
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6707 - accuracy: 0.4997
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6721 - accuracy: 0.4996
22528/25000 [==========================>...] - ETA: 9s - loss: 7.6727 - accuracy: 0.4996
22560/25000 [==========================>...] - ETA: 9s - loss: 7.6707 - accuracy: 0.4997
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6693 - accuracy: 0.4998
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6714 - accuracy: 0.4997
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6720 - accuracy: 0.4996
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6720 - accuracy: 0.4996
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6720 - accuracy: 0.4996
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6727 - accuracy: 0.4996
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6713 - accuracy: 0.4997
22816/25000 [==========================>...] - ETA: 8s - loss: 7.6707 - accuracy: 0.4997
22848/25000 [==========================>...] - ETA: 8s - loss: 7.6693 - accuracy: 0.4998
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6733 - accuracy: 0.4996
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6747 - accuracy: 0.4995
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6766 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6733 - accuracy: 0.4996
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6733 - accuracy: 0.4996
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6746 - accuracy: 0.4995
23072/25000 [==========================>...] - ETA: 7s - loss: 7.6753 - accuracy: 0.4994
23104/25000 [==========================>...] - ETA: 7s - loss: 7.6786 - accuracy: 0.4992
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6819 - accuracy: 0.4990
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6838 - accuracy: 0.4989
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6825 - accuracy: 0.4990
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6831 - accuracy: 0.4989
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6805 - accuracy: 0.4991
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6765 - accuracy: 0.4994
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6778 - accuracy: 0.4993
23360/25000 [===========================>..] - ETA: 6s - loss: 7.6765 - accuracy: 0.4994
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6784 - accuracy: 0.4992
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6764 - accuracy: 0.4994
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6777 - accuracy: 0.4993
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6751 - accuracy: 0.4994
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6757 - accuracy: 0.4994
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6783 - accuracy: 0.4992
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6777 - accuracy: 0.4993
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6796 - accuracy: 0.4992
23648/25000 [===========================>..] - ETA: 5s - loss: 7.6783 - accuracy: 0.4992
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6770 - accuracy: 0.4993
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6724 - accuracy: 0.4996
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6711 - accuracy: 0.4997
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6711 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6724 - accuracy: 0.4996
23904/25000 [===========================>..] - ETA: 4s - loss: 7.6724 - accuracy: 0.4996
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
24192/25000 [============================>.] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24288/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24320/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24352/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24384/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24416/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24448/25000 [============================>.] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24480/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24608/25000 [============================>.] - ETA: 1s - loss: 7.6623 - accuracy: 0.5003
24640/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 1s - loss: 7.6592 - accuracy: 0.5005
24704/25000 [============================>.] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24768/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24800/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24832/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 112s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

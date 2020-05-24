
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/e7305653d03d43cd8d7df2199fe3ac5f94ef1c05', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'e7305653d03d43cd8d7df2199fe3ac5f94ef1c05', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/e7305653d03d43cd8d7df2199fe3ac5f94ef1c05

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/e7305653d03d43cd8d7df2199fe3ac5f94ef1c05

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/e7305653d03d43cd8d7df2199fe3ac5f94ef1c05

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
	Data preprocessing and feature engineering runtime = 0.27s ...
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:59<01:28, 29.51s/it] 40%|████      | 2/5 [00:59<01:28, 29.51s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.4598119535336652, 'embedding_size_factor': 0.9661837275409131, 'layers.choice': 1, 'learning_rate': 0.00016225896633767384, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 7.8145740147869e-11} and reward: 0.3778
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xddm\x8f\x1d\xaf*jX\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\xea\xfa"\xf6\xe6*X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?%D\x81\xe8e\xfb\xd5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xd5{\x04\x81=U\x97u.' and reward: 0.3778
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xddm\x8f\x1d\xaf*jX\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\xea\xfa"\xf6\xe6*X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?%D\x81\xe8e\xfb\xd5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xd5{\x04\x81=U\x97u.' and reward: 0.3778
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 161.94368481636047
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the -43.97s of remaining time.
Ensemble size: 28
Ensemble weights: 
[0.60714286 0.39285714]
	0.3912	 = Validation accuracy score
	1.0s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 165.01s ...
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
 3219456/17464789 [====>.........................] - ETA: 0s
10362880/17464789 [================>.............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-24 00:38:16.911691: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-24 00:38:16.916309: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-24 00:38:16.916468: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5568412a4b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 00:38:16.916486: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:31 - loss: 8.6249 - accuracy: 0.4375
   64/25000 [..............................] - ETA: 2:56 - loss: 8.1458 - accuracy: 0.4688
   96/25000 [..............................] - ETA: 2:23 - loss: 7.9861 - accuracy: 0.4792
  128/25000 [..............................] - ETA: 2:06 - loss: 8.2656 - accuracy: 0.4609
  160/25000 [..............................] - ETA: 1:56 - loss: 7.8583 - accuracy: 0.4875
  192/25000 [..............................] - ETA: 1:49 - loss: 7.7465 - accuracy: 0.4948
  224/25000 [..............................] - ETA: 1:44 - loss: 7.6666 - accuracy: 0.5000
  256/25000 [..............................] - ETA: 1:42 - loss: 7.7864 - accuracy: 0.4922
  288/25000 [..............................] - ETA: 1:40 - loss: 8.0925 - accuracy: 0.4722
  320/25000 [..............................] - ETA: 1:37 - loss: 8.1458 - accuracy: 0.4688
  352/25000 [..............................] - ETA: 1:35 - loss: 8.0587 - accuracy: 0.4744
  384/25000 [..............................] - ETA: 1:34 - loss: 8.0659 - accuracy: 0.4740
  416/25000 [..............................] - ETA: 1:33 - loss: 8.0352 - accuracy: 0.4760
  448/25000 [..............................] - ETA: 1:32 - loss: 8.0431 - accuracy: 0.4754
  480/25000 [..............................] - ETA: 1:30 - loss: 8.0500 - accuracy: 0.4750
  512/25000 [..............................] - ETA: 1:29 - loss: 7.9661 - accuracy: 0.4805
  544/25000 [..............................] - ETA: 1:29 - loss: 7.8075 - accuracy: 0.4908
  576/25000 [..............................] - ETA: 1:28 - loss: 7.7465 - accuracy: 0.4948
  608/25000 [..............................] - ETA: 1:28 - loss: 7.6666 - accuracy: 0.5000
  640/25000 [..............................] - ETA: 1:27 - loss: 7.6906 - accuracy: 0.4984
  672/25000 [..............................] - ETA: 1:26 - loss: 7.6894 - accuracy: 0.4985
  704/25000 [..............................] - ETA: 1:26 - loss: 7.6884 - accuracy: 0.4986
  736/25000 [..............................] - ETA: 1:26 - loss: 7.7708 - accuracy: 0.4932
  768/25000 [..............................] - ETA: 1:25 - loss: 7.8064 - accuracy: 0.4909
  800/25000 [..............................] - ETA: 1:25 - loss: 7.7433 - accuracy: 0.4950
  832/25000 [..............................] - ETA: 1:25 - loss: 7.7035 - accuracy: 0.4976
  864/25000 [>.............................] - ETA: 1:25 - loss: 7.6311 - accuracy: 0.5023
  896/25000 [>.............................] - ETA: 1:24 - loss: 7.7008 - accuracy: 0.4978
  928/25000 [>.............................] - ETA: 1:24 - loss: 7.7658 - accuracy: 0.4935
  960/25000 [>.............................] - ETA: 1:24 - loss: 7.8423 - accuracy: 0.4885
  992/25000 [>.............................] - ETA: 1:23 - loss: 7.7748 - accuracy: 0.4929
 1024/25000 [>.............................] - ETA: 1:25 - loss: 7.8164 - accuracy: 0.4902
 1056/25000 [>.............................] - ETA: 1:25 - loss: 7.7828 - accuracy: 0.4924
 1088/25000 [>.............................] - ETA: 1:24 - loss: 7.8498 - accuracy: 0.4881
 1120/25000 [>.............................] - ETA: 1:24 - loss: 7.8309 - accuracy: 0.4893
 1152/25000 [>.............................] - ETA: 1:24 - loss: 7.8263 - accuracy: 0.4896
 1184/25000 [>.............................] - ETA: 1:24 - loss: 7.8609 - accuracy: 0.4873
 1216/25000 [>.............................] - ETA: 1:23 - loss: 7.8558 - accuracy: 0.4877
 1248/25000 [>.............................] - ETA: 1:23 - loss: 7.8263 - accuracy: 0.4896
 1280/25000 [>.............................] - ETA: 1:23 - loss: 7.7864 - accuracy: 0.4922
 1312/25000 [>.............................] - ETA: 1:22 - loss: 7.8185 - accuracy: 0.4901
 1344/25000 [>.............................] - ETA: 1:22 - loss: 7.8606 - accuracy: 0.4874
 1376/25000 [>.............................] - ETA: 1:22 - loss: 7.8003 - accuracy: 0.4913
 1408/25000 [>.............................] - ETA: 1:21 - loss: 7.8409 - accuracy: 0.4886
 1440/25000 [>.............................] - ETA: 1:21 - loss: 7.8583 - accuracy: 0.4875
 1472/25000 [>.............................] - ETA: 1:21 - loss: 7.8750 - accuracy: 0.4864
 1504/25000 [>.............................] - ETA: 1:21 - loss: 7.8705 - accuracy: 0.4867
 1536/25000 [>.............................] - ETA: 1:20 - loss: 7.8164 - accuracy: 0.4902
 1568/25000 [>.............................] - ETA: 1:20 - loss: 7.7742 - accuracy: 0.4930
 1600/25000 [>.............................] - ETA: 1:20 - loss: 7.7816 - accuracy: 0.4925
 1632/25000 [>.............................] - ETA: 1:20 - loss: 7.8263 - accuracy: 0.4896
 1664/25000 [>.............................] - ETA: 1:19 - loss: 7.8325 - accuracy: 0.4892
 1696/25000 [=>............................] - ETA: 1:19 - loss: 7.8203 - accuracy: 0.4900
 1728/25000 [=>............................] - ETA: 1:19 - loss: 7.8441 - accuracy: 0.4884
 1760/25000 [=>............................] - ETA: 1:19 - loss: 7.7973 - accuracy: 0.4915
 1792/25000 [=>............................] - ETA: 1:19 - loss: 7.7864 - accuracy: 0.4922
 1824/25000 [=>............................] - ETA: 1:19 - loss: 7.7675 - accuracy: 0.4934
 1856/25000 [=>............................] - ETA: 1:18 - loss: 7.7410 - accuracy: 0.4952
 1888/25000 [=>............................] - ETA: 1:18 - loss: 7.7560 - accuracy: 0.4942
 1920/25000 [=>............................] - ETA: 1:18 - loss: 7.7625 - accuracy: 0.4938
 1952/25000 [=>............................] - ETA: 1:18 - loss: 7.7687 - accuracy: 0.4933
 1984/25000 [=>............................] - ETA: 1:18 - loss: 7.7594 - accuracy: 0.4940
 2016/25000 [=>............................] - ETA: 1:18 - loss: 7.7351 - accuracy: 0.4955
 2048/25000 [=>............................] - ETA: 1:17 - loss: 7.7265 - accuracy: 0.4961
 2080/25000 [=>............................] - ETA: 1:17 - loss: 7.7182 - accuracy: 0.4966
 2112/25000 [=>............................] - ETA: 1:17 - loss: 7.7465 - accuracy: 0.4948
 2144/25000 [=>............................] - ETA: 1:17 - loss: 7.7596 - accuracy: 0.4939
 2176/25000 [=>............................] - ETA: 1:17 - loss: 7.8005 - accuracy: 0.4913
 2208/25000 [=>............................] - ETA: 1:17 - loss: 7.7986 - accuracy: 0.4914
 2240/25000 [=>............................] - ETA: 1:17 - loss: 7.7967 - accuracy: 0.4915
 2272/25000 [=>............................] - ETA: 1:16 - loss: 7.7813 - accuracy: 0.4925
 2304/25000 [=>............................] - ETA: 1:16 - loss: 7.7598 - accuracy: 0.4939
 2336/25000 [=>............................] - ETA: 1:16 - loss: 7.7454 - accuracy: 0.4949
 2368/25000 [=>............................] - ETA: 1:16 - loss: 7.7767 - accuracy: 0.4928
 2400/25000 [=>............................] - ETA: 1:16 - loss: 7.8008 - accuracy: 0.4913
 2432/25000 [=>............................] - ETA: 1:16 - loss: 7.7864 - accuracy: 0.4922
 2464/25000 [=>............................] - ETA: 1:16 - loss: 7.7973 - accuracy: 0.4915
 2496/25000 [=>............................] - ETA: 1:15 - loss: 7.7833 - accuracy: 0.4924
 2528/25000 [==>...........................] - ETA: 1:15 - loss: 7.7819 - accuracy: 0.4925
 2560/25000 [==>...........................] - ETA: 1:15 - loss: 7.7924 - accuracy: 0.4918
 2592/25000 [==>...........................] - ETA: 1:15 - loss: 7.8027 - accuracy: 0.4911
 2624/25000 [==>...........................] - ETA: 1:15 - loss: 7.8010 - accuracy: 0.4912
 2656/25000 [==>...........................] - ETA: 1:15 - loss: 7.8225 - accuracy: 0.4898
 2688/25000 [==>...........................] - ETA: 1:14 - loss: 7.8377 - accuracy: 0.4888
 2720/25000 [==>...........................] - ETA: 1:14 - loss: 7.8245 - accuracy: 0.4897
 2752/25000 [==>...........................] - ETA: 1:14 - loss: 7.8115 - accuracy: 0.4906
 2784/25000 [==>...........................] - ETA: 1:14 - loss: 7.8043 - accuracy: 0.4910
 2816/25000 [==>...........................] - ETA: 1:14 - loss: 7.8245 - accuracy: 0.4897
 2848/25000 [==>...........................] - ETA: 1:14 - loss: 7.8120 - accuracy: 0.4905
 2880/25000 [==>...........................] - ETA: 1:13 - loss: 7.7944 - accuracy: 0.4917
 2912/25000 [==>...........................] - ETA: 1:13 - loss: 7.7719 - accuracy: 0.4931
 2944/25000 [==>...........................] - ETA: 1:13 - loss: 7.7760 - accuracy: 0.4929
 2976/25000 [==>...........................] - ETA: 1:13 - loss: 7.7594 - accuracy: 0.4940
 3008/25000 [==>...........................] - ETA: 1:13 - loss: 7.7533 - accuracy: 0.4943
 3040/25000 [==>...........................] - ETA: 1:13 - loss: 7.7524 - accuracy: 0.4944
 3072/25000 [==>...........................] - ETA: 1:12 - loss: 7.7365 - accuracy: 0.4954
 3104/25000 [==>...........................] - ETA: 1:12 - loss: 7.7358 - accuracy: 0.4955
 3136/25000 [==>...........................] - ETA: 1:12 - loss: 7.7204 - accuracy: 0.4965
 3168/25000 [==>...........................] - ETA: 1:12 - loss: 7.7053 - accuracy: 0.4975
 3200/25000 [==>...........................] - ETA: 1:12 - loss: 7.6906 - accuracy: 0.4984
 3232/25000 [==>...........................] - ETA: 1:12 - loss: 7.6998 - accuracy: 0.4978
 3264/25000 [==>...........................] - ETA: 1:12 - loss: 7.7042 - accuracy: 0.4975
 3296/25000 [==>...........................] - ETA: 1:12 - loss: 7.7085 - accuracy: 0.4973
 3328/25000 [==>...........................] - ETA: 1:12 - loss: 7.7173 - accuracy: 0.4967
 3360/25000 [===>..........................] - ETA: 1:11 - loss: 7.7031 - accuracy: 0.4976
 3392/25000 [===>..........................] - ETA: 1:11 - loss: 7.7118 - accuracy: 0.4971
 3424/25000 [===>..........................] - ETA: 1:11 - loss: 7.7024 - accuracy: 0.4977
 3456/25000 [===>..........................] - ETA: 1:11 - loss: 7.7110 - accuracy: 0.4971
 3488/25000 [===>..........................] - ETA: 1:11 - loss: 7.6974 - accuracy: 0.4980
 3520/25000 [===>..........................] - ETA: 1:11 - loss: 7.6797 - accuracy: 0.4991
 3552/25000 [===>..........................] - ETA: 1:11 - loss: 7.6968 - accuracy: 0.4980
 3584/25000 [===>..........................] - ETA: 1:11 - loss: 7.6966 - accuracy: 0.4980
 3616/25000 [===>..........................] - ETA: 1:11 - loss: 7.7217 - accuracy: 0.4964
 3648/25000 [===>..........................] - ETA: 1:10 - loss: 7.7002 - accuracy: 0.4978
 3680/25000 [===>..........................] - ETA: 1:10 - loss: 7.6958 - accuracy: 0.4981
 3712/25000 [===>..........................] - ETA: 1:10 - loss: 7.7038 - accuracy: 0.4976
 3744/25000 [===>..........................] - ETA: 1:10 - loss: 7.7321 - accuracy: 0.4957
 3776/25000 [===>..........................] - ETA: 1:10 - loss: 7.7235 - accuracy: 0.4963
 3808/25000 [===>..........................] - ETA: 1:10 - loss: 7.7310 - accuracy: 0.4958
 3840/25000 [===>..........................] - ETA: 1:10 - loss: 7.7585 - accuracy: 0.4940
 3872/25000 [===>..........................] - ETA: 1:10 - loss: 7.7815 - accuracy: 0.4925
 3904/25000 [===>..........................] - ETA: 1:09 - loss: 7.7687 - accuracy: 0.4933
 3936/25000 [===>..........................] - ETA: 1:09 - loss: 7.7718 - accuracy: 0.4931
 3968/25000 [===>..........................] - ETA: 1:09 - loss: 7.7632 - accuracy: 0.4937
 4000/25000 [===>..........................] - ETA: 1:09 - loss: 7.7663 - accuracy: 0.4935
 4032/25000 [===>..........................] - ETA: 1:09 - loss: 7.7655 - accuracy: 0.4936
 4064/25000 [===>..........................] - ETA: 1:09 - loss: 7.7685 - accuracy: 0.4934
 4096/25000 [===>..........................] - ETA: 1:09 - loss: 7.7677 - accuracy: 0.4934
 4128/25000 [===>..........................] - ETA: 1:09 - loss: 7.7669 - accuracy: 0.4935
 4160/25000 [===>..........................] - ETA: 1:09 - loss: 7.7846 - accuracy: 0.4923
 4192/25000 [====>.........................] - ETA: 1:09 - loss: 7.7837 - accuracy: 0.4924
 4224/25000 [====>.........................] - ETA: 1:08 - loss: 7.7937 - accuracy: 0.4917
 4256/25000 [====>.........................] - ETA: 1:08 - loss: 7.7999 - accuracy: 0.4913
 4288/25000 [====>.........................] - ETA: 1:08 - loss: 7.7989 - accuracy: 0.4914
 4320/25000 [====>.........................] - ETA: 1:08 - loss: 7.7873 - accuracy: 0.4921
 4352/25000 [====>.........................] - ETA: 1:08 - loss: 7.7935 - accuracy: 0.4917
 4384/25000 [====>.........................] - ETA: 1:08 - loss: 7.7855 - accuracy: 0.4922
 4416/25000 [====>.........................] - ETA: 1:08 - loss: 7.7604 - accuracy: 0.4939
 4448/25000 [====>.........................] - ETA: 1:08 - loss: 7.7562 - accuracy: 0.4942
 4480/25000 [====>.........................] - ETA: 1:07 - loss: 7.7488 - accuracy: 0.4946
 4512/25000 [====>.........................] - ETA: 1:07 - loss: 7.7516 - accuracy: 0.4945
 4544/25000 [====>.........................] - ETA: 1:07 - loss: 7.7510 - accuracy: 0.4945
 4576/25000 [====>.........................] - ETA: 1:07 - loss: 7.7437 - accuracy: 0.4950
 4608/25000 [====>.........................] - ETA: 1:07 - loss: 7.7465 - accuracy: 0.4948
 4640/25000 [====>.........................] - ETA: 1:07 - loss: 7.7591 - accuracy: 0.4940
 4672/25000 [====>.........................] - ETA: 1:07 - loss: 7.7618 - accuracy: 0.4938
 4704/25000 [====>.........................] - ETA: 1:07 - loss: 7.7677 - accuracy: 0.4934
 4736/25000 [====>.........................] - ETA: 1:06 - loss: 7.7605 - accuracy: 0.4939
 4768/25000 [====>.........................] - ETA: 1:06 - loss: 7.7663 - accuracy: 0.4935
 4800/25000 [====>.........................] - ETA: 1:06 - loss: 7.7688 - accuracy: 0.4933
 4832/25000 [====>.........................] - ETA: 1:06 - loss: 7.7745 - accuracy: 0.4930
 4864/25000 [====>.........................] - ETA: 1:06 - loss: 7.7643 - accuracy: 0.4936
 4896/25000 [====>.........................] - ETA: 1:06 - loss: 7.7731 - accuracy: 0.4931
 4928/25000 [====>.........................] - ETA: 1:06 - loss: 7.7569 - accuracy: 0.4941
 4960/25000 [====>.........................] - ETA: 1:06 - loss: 7.7501 - accuracy: 0.4946
 4992/25000 [====>.........................] - ETA: 1:06 - loss: 7.7465 - accuracy: 0.4948
 5024/25000 [=====>........................] - ETA: 1:06 - loss: 7.7521 - accuracy: 0.4944
 5056/25000 [=====>........................] - ETA: 1:05 - loss: 7.7606 - accuracy: 0.4939
 5088/25000 [=====>........................] - ETA: 1:05 - loss: 7.7510 - accuracy: 0.4945
 5120/25000 [=====>........................] - ETA: 1:05 - loss: 7.7385 - accuracy: 0.4953
 5152/25000 [=====>........................] - ETA: 1:05 - loss: 7.7440 - accuracy: 0.4950
 5184/25000 [=====>........................] - ETA: 1:05 - loss: 7.7376 - accuracy: 0.4954
 5216/25000 [=====>........................] - ETA: 1:05 - loss: 7.7489 - accuracy: 0.4946
 5248/25000 [=====>........................] - ETA: 1:05 - loss: 7.7513 - accuracy: 0.4945
 5280/25000 [=====>........................] - ETA: 1:05 - loss: 7.7508 - accuracy: 0.4945
 5312/25000 [=====>........................] - ETA: 1:05 - loss: 7.7532 - accuracy: 0.4944
 5344/25000 [=====>........................] - ETA: 1:04 - loss: 7.7441 - accuracy: 0.4949
 5376/25000 [=====>........................] - ETA: 1:04 - loss: 7.7465 - accuracy: 0.4948
 5408/25000 [=====>........................] - ETA: 1:04 - loss: 7.7347 - accuracy: 0.4956
 5440/25000 [=====>........................] - ETA: 1:04 - loss: 7.7343 - accuracy: 0.4956
 5472/25000 [=====>........................] - ETA: 1:04 - loss: 7.7339 - accuracy: 0.4956
 5504/25000 [=====>........................] - ETA: 1:04 - loss: 7.7251 - accuracy: 0.4962
 5536/25000 [=====>........................] - ETA: 1:04 - loss: 7.7303 - accuracy: 0.4958
 5568/25000 [=====>........................] - ETA: 1:04 - loss: 7.7327 - accuracy: 0.4957
 5600/25000 [=====>........................] - ETA: 1:04 - loss: 7.7351 - accuracy: 0.4955
 5632/25000 [=====>........................] - ETA: 1:03 - loss: 7.7265 - accuracy: 0.4961
 5664/25000 [=====>........................] - ETA: 1:03 - loss: 7.7126 - accuracy: 0.4970
 5696/25000 [=====>........................] - ETA: 1:03 - loss: 7.7178 - accuracy: 0.4967
 5728/25000 [=====>........................] - ETA: 1:03 - loss: 7.7148 - accuracy: 0.4969
 5760/25000 [=====>........................] - ETA: 1:03 - loss: 7.7172 - accuracy: 0.4967
 5792/25000 [=====>........................] - ETA: 1:03 - loss: 7.7037 - accuracy: 0.4976
 5824/25000 [=====>........................] - ETA: 1:03 - loss: 7.7035 - accuracy: 0.4976
 5856/25000 [======>.......................] - ETA: 1:03 - loss: 7.6980 - accuracy: 0.4980
 5888/25000 [======>.......................] - ETA: 1:03 - loss: 7.6822 - accuracy: 0.4990
 5920/25000 [======>.......................] - ETA: 1:02 - loss: 7.6822 - accuracy: 0.4990
 5952/25000 [======>.......................] - ETA: 1:02 - loss: 7.6847 - accuracy: 0.4988
 5984/25000 [======>.......................] - ETA: 1:02 - loss: 7.6897 - accuracy: 0.4985
 6016/25000 [======>.......................] - ETA: 1:02 - loss: 7.6921 - accuracy: 0.4983
 6048/25000 [======>.......................] - ETA: 1:02 - loss: 7.6945 - accuracy: 0.4982
 6080/25000 [======>.......................] - ETA: 1:02 - loss: 7.6918 - accuracy: 0.4984
 6112/25000 [======>.......................] - ETA: 1:02 - loss: 7.6917 - accuracy: 0.4984
 6144/25000 [======>.......................] - ETA: 1:02 - loss: 7.6966 - accuracy: 0.4980
 6176/25000 [======>.......................] - ETA: 1:02 - loss: 7.6964 - accuracy: 0.4981
 6208/25000 [======>.......................] - ETA: 1:02 - loss: 7.7012 - accuracy: 0.4977
 6240/25000 [======>.......................] - ETA: 1:01 - loss: 7.6986 - accuracy: 0.4979
 6272/25000 [======>.......................] - ETA: 1:01 - loss: 7.7008 - accuracy: 0.4978
 6304/25000 [======>.......................] - ETA: 1:01 - loss: 7.7080 - accuracy: 0.4973
 6336/25000 [======>.......................] - ETA: 1:01 - loss: 7.7150 - accuracy: 0.4968
 6368/25000 [======>.......................] - ETA: 1:01 - loss: 7.7172 - accuracy: 0.4967
 6400/25000 [======>.......................] - ETA: 1:01 - loss: 7.7241 - accuracy: 0.4963
 6432/25000 [======>.......................] - ETA: 1:01 - loss: 7.7262 - accuracy: 0.4961
 6464/25000 [======>.......................] - ETA: 1:01 - loss: 7.7307 - accuracy: 0.4958
 6496/25000 [======>.......................] - ETA: 1:01 - loss: 7.7280 - accuracy: 0.4960
 6528/25000 [======>.......................] - ETA: 1:00 - loss: 7.7253 - accuracy: 0.4962
 6560/25000 [======>.......................] - ETA: 1:00 - loss: 7.7274 - accuracy: 0.4960
 6592/25000 [======>.......................] - ETA: 1:00 - loss: 7.7271 - accuracy: 0.4961
 6624/25000 [======>.......................] - ETA: 1:00 - loss: 7.7245 - accuracy: 0.4962
 6656/25000 [======>.......................] - ETA: 1:00 - loss: 7.7173 - accuracy: 0.4967
 6688/25000 [=======>......................] - ETA: 1:00 - loss: 7.7216 - accuracy: 0.4964
 6720/25000 [=======>......................] - ETA: 1:00 - loss: 7.7168 - accuracy: 0.4967
 6752/25000 [=======>......................] - ETA: 1:00 - loss: 7.7211 - accuracy: 0.4964
 6784/25000 [=======>......................] - ETA: 59s - loss: 7.7209 - accuracy: 0.4965 
 6816/25000 [=======>......................] - ETA: 59s - loss: 7.7206 - accuracy: 0.4965
 6848/25000 [=======>......................] - ETA: 59s - loss: 7.7159 - accuracy: 0.4968
 6880/25000 [=======>......................] - ETA: 59s - loss: 7.7157 - accuracy: 0.4968
 6912/25000 [=======>......................] - ETA: 59s - loss: 7.7154 - accuracy: 0.4968
 6944/25000 [=======>......................] - ETA: 59s - loss: 7.7218 - accuracy: 0.4964
 6976/25000 [=======>......................] - ETA: 59s - loss: 7.7194 - accuracy: 0.4966
 7008/25000 [=======>......................] - ETA: 59s - loss: 7.7191 - accuracy: 0.4966
 7040/25000 [=======>......................] - ETA: 59s - loss: 7.7298 - accuracy: 0.4959
 7072/25000 [=======>......................] - ETA: 58s - loss: 7.7317 - accuracy: 0.4958
 7104/25000 [=======>......................] - ETA: 58s - loss: 7.7378 - accuracy: 0.4954
 7136/25000 [=======>......................] - ETA: 58s - loss: 7.7375 - accuracy: 0.4954
 7168/25000 [=======>......................] - ETA: 58s - loss: 7.7351 - accuracy: 0.4955
 7200/25000 [=======>......................] - ETA: 58s - loss: 7.7433 - accuracy: 0.4950
 7232/25000 [=======>......................] - ETA: 58s - loss: 7.7387 - accuracy: 0.4953
 7264/25000 [=======>......................] - ETA: 58s - loss: 7.7321 - accuracy: 0.4957
 7296/25000 [=======>......................] - ETA: 58s - loss: 7.7318 - accuracy: 0.4958
 7328/25000 [=======>......................] - ETA: 58s - loss: 7.7294 - accuracy: 0.4959
 7360/25000 [=======>......................] - ETA: 57s - loss: 7.7208 - accuracy: 0.4965
 7392/25000 [=======>......................] - ETA: 57s - loss: 7.7164 - accuracy: 0.4968
 7424/25000 [=======>......................] - ETA: 57s - loss: 7.7162 - accuracy: 0.4968
 7456/25000 [=======>......................] - ETA: 57s - loss: 7.7119 - accuracy: 0.4970
 7488/25000 [=======>......................] - ETA: 57s - loss: 7.7096 - accuracy: 0.4972
 7520/25000 [========>.....................] - ETA: 57s - loss: 7.7196 - accuracy: 0.4965
 7552/25000 [========>.....................] - ETA: 57s - loss: 7.7214 - accuracy: 0.4964
 7584/25000 [========>.....................] - ETA: 57s - loss: 7.7212 - accuracy: 0.4964
 7616/25000 [========>.....................] - ETA: 57s - loss: 7.7210 - accuracy: 0.4965
 7648/25000 [========>.....................] - ETA: 56s - loss: 7.7248 - accuracy: 0.4962
 7680/25000 [========>.....................] - ETA: 56s - loss: 7.7285 - accuracy: 0.4960
 7712/25000 [========>.....................] - ETA: 56s - loss: 7.7263 - accuracy: 0.4961
 7744/25000 [========>.....................] - ETA: 56s - loss: 7.7339 - accuracy: 0.4956
 7776/25000 [========>.....................] - ETA: 56s - loss: 7.7337 - accuracy: 0.4956
 7808/25000 [========>.....................] - ETA: 56s - loss: 7.7275 - accuracy: 0.4960
 7840/25000 [========>.....................] - ETA: 56s - loss: 7.7253 - accuracy: 0.4962
 7872/25000 [========>.....................] - ETA: 56s - loss: 7.7309 - accuracy: 0.4958
 7904/25000 [========>.....................] - ETA: 56s - loss: 7.7326 - accuracy: 0.4957
 7936/25000 [========>.....................] - ETA: 56s - loss: 7.7381 - accuracy: 0.4953
 7968/25000 [========>.....................] - ETA: 55s - loss: 7.7397 - accuracy: 0.4952
 8000/25000 [========>.....................] - ETA: 55s - loss: 7.7414 - accuracy: 0.4951
 8032/25000 [========>.....................] - ETA: 55s - loss: 7.7487 - accuracy: 0.4946
 8064/25000 [========>.....................] - ETA: 55s - loss: 7.7465 - accuracy: 0.4948
 8096/25000 [========>.....................] - ETA: 55s - loss: 7.7424 - accuracy: 0.4951
 8128/25000 [========>.....................] - ETA: 55s - loss: 7.7308 - accuracy: 0.4958
 8160/25000 [========>.....................] - ETA: 55s - loss: 7.7343 - accuracy: 0.4956
 8192/25000 [========>.....................] - ETA: 55s - loss: 7.7434 - accuracy: 0.4950
 8224/25000 [========>.....................] - ETA: 55s - loss: 7.7524 - accuracy: 0.4944
 8256/25000 [========>.....................] - ETA: 54s - loss: 7.7539 - accuracy: 0.4943
 8288/25000 [========>.....................] - ETA: 54s - loss: 7.7536 - accuracy: 0.4943
 8320/25000 [========>.....................] - ETA: 54s - loss: 7.7606 - accuracy: 0.4939
 8352/25000 [=========>....................] - ETA: 54s - loss: 7.7602 - accuracy: 0.4939
 8384/25000 [=========>....................] - ETA: 54s - loss: 7.7562 - accuracy: 0.4942
 8416/25000 [=========>....................] - ETA: 54s - loss: 7.7541 - accuracy: 0.4943
 8448/25000 [=========>....................] - ETA: 54s - loss: 7.7483 - accuracy: 0.4947
 8480/25000 [=========>....................] - ETA: 54s - loss: 7.7552 - accuracy: 0.4942
 8512/25000 [=========>....................] - ETA: 54s - loss: 7.7531 - accuracy: 0.4944
 8544/25000 [=========>....................] - ETA: 53s - loss: 7.7456 - accuracy: 0.4949
 8576/25000 [=========>....................] - ETA: 53s - loss: 7.7507 - accuracy: 0.4945
 8608/25000 [=========>....................] - ETA: 53s - loss: 7.7486 - accuracy: 0.4947
 8640/25000 [=========>....................] - ETA: 53s - loss: 7.7518 - accuracy: 0.4944
 8672/25000 [=========>....................] - ETA: 53s - loss: 7.7603 - accuracy: 0.4939
 8704/25000 [=========>....................] - ETA: 53s - loss: 7.7600 - accuracy: 0.4939
 8736/25000 [=========>....................] - ETA: 53s - loss: 7.7614 - accuracy: 0.4938
 8768/25000 [=========>....................] - ETA: 53s - loss: 7.7541 - accuracy: 0.4943
 8800/25000 [=========>....................] - ETA: 53s - loss: 7.7468 - accuracy: 0.4948
 8832/25000 [=========>....................] - ETA: 53s - loss: 7.7500 - accuracy: 0.4946
 8864/25000 [=========>....................] - ETA: 52s - loss: 7.7531 - accuracy: 0.4944
 8896/25000 [=========>....................] - ETA: 52s - loss: 7.7597 - accuracy: 0.4939
 8928/25000 [=========>....................] - ETA: 52s - loss: 7.7662 - accuracy: 0.4935
 8960/25000 [=========>....................] - ETA: 52s - loss: 7.7676 - accuracy: 0.4934
 8992/25000 [=========>....................] - ETA: 52s - loss: 7.7672 - accuracy: 0.4934
 9024/25000 [=========>....................] - ETA: 52s - loss: 7.7601 - accuracy: 0.4939
 9056/25000 [=========>....................] - ETA: 52s - loss: 7.7513 - accuracy: 0.4945
 9088/25000 [=========>....................] - ETA: 52s - loss: 7.7527 - accuracy: 0.4944
 9120/25000 [=========>....................] - ETA: 52s - loss: 7.7557 - accuracy: 0.4942
 9152/25000 [=========>....................] - ETA: 51s - loss: 7.7571 - accuracy: 0.4941
 9184/25000 [==========>...................] - ETA: 51s - loss: 7.7568 - accuracy: 0.4941
 9216/25000 [==========>...................] - ETA: 51s - loss: 7.7515 - accuracy: 0.4945
 9248/25000 [==========>...................] - ETA: 51s - loss: 7.7445 - accuracy: 0.4949
 9280/25000 [==========>...................] - ETA: 51s - loss: 7.7426 - accuracy: 0.4950
 9312/25000 [==========>...................] - ETA: 51s - loss: 7.7473 - accuracy: 0.4947
 9344/25000 [==========>...................] - ETA: 51s - loss: 7.7437 - accuracy: 0.4950
 9376/25000 [==========>...................] - ETA: 51s - loss: 7.7500 - accuracy: 0.4946
 9408/25000 [==========>...................] - ETA: 51s - loss: 7.7481 - accuracy: 0.4947
 9440/25000 [==========>...................] - ETA: 50s - loss: 7.7495 - accuracy: 0.4946
 9472/25000 [==========>...................] - ETA: 50s - loss: 7.7508 - accuracy: 0.4945
 9504/25000 [==========>...................] - ETA: 50s - loss: 7.7570 - accuracy: 0.4941
 9536/25000 [==========>...................] - ETA: 50s - loss: 7.7599 - accuracy: 0.4939
 9568/25000 [==========>...................] - ETA: 50s - loss: 7.7548 - accuracy: 0.4943
 9600/25000 [==========>...................] - ETA: 50s - loss: 7.7481 - accuracy: 0.4947
 9632/25000 [==========>...................] - ETA: 50s - loss: 7.7446 - accuracy: 0.4949
 9664/25000 [==========>...................] - ETA: 50s - loss: 7.7396 - accuracy: 0.4952
 9696/25000 [==========>...................] - ETA: 50s - loss: 7.7378 - accuracy: 0.4954
 9728/25000 [==========>...................] - ETA: 50s - loss: 7.7344 - accuracy: 0.4956
 9760/25000 [==========>...................] - ETA: 49s - loss: 7.7357 - accuracy: 0.4955
 9792/25000 [==========>...................] - ETA: 49s - loss: 7.7340 - accuracy: 0.4956
 9824/25000 [==========>...................] - ETA: 49s - loss: 7.7259 - accuracy: 0.4961
 9856/25000 [==========>...................] - ETA: 49s - loss: 7.7288 - accuracy: 0.4959
 9888/25000 [==========>...................] - ETA: 49s - loss: 7.7224 - accuracy: 0.4964
 9920/25000 [==========>...................] - ETA: 49s - loss: 7.7176 - accuracy: 0.4967
 9952/25000 [==========>...................] - ETA: 49s - loss: 7.7190 - accuracy: 0.4966
 9984/25000 [==========>...................] - ETA: 49s - loss: 7.7204 - accuracy: 0.4965
10016/25000 [===========>..................] - ETA: 49s - loss: 7.7248 - accuracy: 0.4962
10048/25000 [===========>..................] - ETA: 49s - loss: 7.7261 - accuracy: 0.4961
10080/25000 [===========>..................] - ETA: 48s - loss: 7.7244 - accuracy: 0.4962
10112/25000 [===========>..................] - ETA: 48s - loss: 7.7242 - accuracy: 0.4962
10144/25000 [===========>..................] - ETA: 48s - loss: 7.7241 - accuracy: 0.4963
10176/25000 [===========>..................] - ETA: 48s - loss: 7.7254 - accuracy: 0.4962
10208/25000 [===========>..................] - ETA: 48s - loss: 7.7252 - accuracy: 0.4962
10240/25000 [===========>..................] - ETA: 48s - loss: 7.7295 - accuracy: 0.4959
10272/25000 [===========>..................] - ETA: 48s - loss: 7.7308 - accuracy: 0.4958
10304/25000 [===========>..................] - ETA: 48s - loss: 7.7247 - accuracy: 0.4962
10336/25000 [===========>..................] - ETA: 48s - loss: 7.7304 - accuracy: 0.4958
10368/25000 [===========>..................] - ETA: 47s - loss: 7.7287 - accuracy: 0.4959
10400/25000 [===========>..................] - ETA: 47s - loss: 7.7330 - accuracy: 0.4957
10432/25000 [===========>..................] - ETA: 47s - loss: 7.7328 - accuracy: 0.4957
10464/25000 [===========>..................] - ETA: 47s - loss: 7.7384 - accuracy: 0.4953
10496/25000 [===========>..................] - ETA: 47s - loss: 7.7367 - accuracy: 0.4954
10528/25000 [===========>..................] - ETA: 47s - loss: 7.7394 - accuracy: 0.4953
10560/25000 [===========>..................] - ETA: 47s - loss: 7.7334 - accuracy: 0.4956
10592/25000 [===========>..................] - ETA: 47s - loss: 7.7274 - accuracy: 0.4960
10624/25000 [===========>..................] - ETA: 47s - loss: 7.7345 - accuracy: 0.4956
10656/25000 [===========>..................] - ETA: 47s - loss: 7.7328 - accuracy: 0.4957
10688/25000 [===========>..................] - ETA: 46s - loss: 7.7283 - accuracy: 0.4960
10720/25000 [===========>..................] - ETA: 46s - loss: 7.7296 - accuracy: 0.4959
10752/25000 [===========>..................] - ETA: 46s - loss: 7.7308 - accuracy: 0.4958
10784/25000 [===========>..................] - ETA: 46s - loss: 7.7206 - accuracy: 0.4965
10816/25000 [===========>..................] - ETA: 46s - loss: 7.7191 - accuracy: 0.4966
10848/25000 [============>.................] - ETA: 46s - loss: 7.7175 - accuracy: 0.4967
10880/25000 [============>.................] - ETA: 46s - loss: 7.7174 - accuracy: 0.4967
10912/25000 [============>.................] - ETA: 46s - loss: 7.7102 - accuracy: 0.4972
10944/25000 [============>.................] - ETA: 46s - loss: 7.7087 - accuracy: 0.4973
10976/25000 [============>.................] - ETA: 45s - loss: 7.7057 - accuracy: 0.4974
11008/25000 [============>.................] - ETA: 45s - loss: 7.7084 - accuracy: 0.4973
11040/25000 [============>.................] - ETA: 45s - loss: 7.7069 - accuracy: 0.4974
11072/25000 [============>.................] - ETA: 45s - loss: 7.7123 - accuracy: 0.4970
11104/25000 [============>.................] - ETA: 45s - loss: 7.7177 - accuracy: 0.4967
11136/25000 [============>.................] - ETA: 45s - loss: 7.7134 - accuracy: 0.4969
11168/25000 [============>.................] - ETA: 45s - loss: 7.7160 - accuracy: 0.4968
11200/25000 [============>.................] - ETA: 45s - loss: 7.7200 - accuracy: 0.4965
11232/25000 [============>.................] - ETA: 45s - loss: 7.7103 - accuracy: 0.4972
11264/25000 [============>.................] - ETA: 44s - loss: 7.7143 - accuracy: 0.4969
11296/25000 [============>.................] - ETA: 44s - loss: 7.7155 - accuracy: 0.4968
11328/25000 [============>.................] - ETA: 44s - loss: 7.7194 - accuracy: 0.4966
11360/25000 [============>.................] - ETA: 44s - loss: 7.7220 - accuracy: 0.4964
11392/25000 [============>.................] - ETA: 44s - loss: 7.7205 - accuracy: 0.4965
11424/25000 [============>.................] - ETA: 44s - loss: 7.7216 - accuracy: 0.4964
11456/25000 [============>.................] - ETA: 44s - loss: 7.7188 - accuracy: 0.4966
11488/25000 [============>.................] - ETA: 44s - loss: 7.7147 - accuracy: 0.4969
11520/25000 [============>.................] - ETA: 44s - loss: 7.7199 - accuracy: 0.4965
11552/25000 [============>.................] - ETA: 43s - loss: 7.7157 - accuracy: 0.4968
11584/25000 [============>.................] - ETA: 43s - loss: 7.7143 - accuracy: 0.4969
11616/25000 [============>.................] - ETA: 43s - loss: 7.7141 - accuracy: 0.4969
11648/25000 [============>.................] - ETA: 43s - loss: 7.7193 - accuracy: 0.4966
11680/25000 [=============>................] - ETA: 43s - loss: 7.7231 - accuracy: 0.4963
11712/25000 [=============>................] - ETA: 43s - loss: 7.7268 - accuracy: 0.4961
11744/25000 [=============>................] - ETA: 43s - loss: 7.7306 - accuracy: 0.4958
11776/25000 [=============>................] - ETA: 43s - loss: 7.7304 - accuracy: 0.4958
11808/25000 [=============>................] - ETA: 43s - loss: 7.7328 - accuracy: 0.4957
11840/25000 [=============>................] - ETA: 42s - loss: 7.7378 - accuracy: 0.4954
11872/25000 [=============>................] - ETA: 42s - loss: 7.7273 - accuracy: 0.4960
11904/25000 [=============>................] - ETA: 42s - loss: 7.7233 - accuracy: 0.4963
11936/25000 [=============>................] - ETA: 42s - loss: 7.7206 - accuracy: 0.4965
11968/25000 [=============>................] - ETA: 42s - loss: 7.7179 - accuracy: 0.4967
12000/25000 [=============>................] - ETA: 42s - loss: 7.7152 - accuracy: 0.4968
12032/25000 [=============>................] - ETA: 42s - loss: 7.7176 - accuracy: 0.4967
12064/25000 [=============>................] - ETA: 42s - loss: 7.7238 - accuracy: 0.4963
12096/25000 [=============>................] - ETA: 42s - loss: 7.7249 - accuracy: 0.4962
12128/25000 [=============>................] - ETA: 42s - loss: 7.7260 - accuracy: 0.4961
12160/25000 [=============>................] - ETA: 41s - loss: 7.7271 - accuracy: 0.4961
12192/25000 [=============>................] - ETA: 41s - loss: 7.7295 - accuracy: 0.4959
12224/25000 [=============>................] - ETA: 41s - loss: 7.7293 - accuracy: 0.4959
12256/25000 [=============>................] - ETA: 41s - loss: 7.7292 - accuracy: 0.4959
12288/25000 [=============>................] - ETA: 41s - loss: 7.7340 - accuracy: 0.4956
12320/25000 [=============>................] - ETA: 41s - loss: 7.7313 - accuracy: 0.4958
12352/25000 [=============>................] - ETA: 41s - loss: 7.7312 - accuracy: 0.4958
12384/25000 [=============>................] - ETA: 41s - loss: 7.7223 - accuracy: 0.4964
12416/25000 [=============>................] - ETA: 41s - loss: 7.7234 - accuracy: 0.4963
12448/25000 [=============>................] - ETA: 40s - loss: 7.7220 - accuracy: 0.4964
12480/25000 [=============>................] - ETA: 40s - loss: 7.7182 - accuracy: 0.4966
12512/25000 [==============>...............] - ETA: 40s - loss: 7.7144 - accuracy: 0.4969
12544/25000 [==============>...............] - ETA: 40s - loss: 7.7094 - accuracy: 0.4972
12576/25000 [==============>...............] - ETA: 40s - loss: 7.7117 - accuracy: 0.4971
12608/25000 [==============>...............] - ETA: 40s - loss: 7.7128 - accuracy: 0.4970
12640/25000 [==============>...............] - ETA: 40s - loss: 7.7139 - accuracy: 0.4969
12672/25000 [==============>...............] - ETA: 40s - loss: 7.7150 - accuracy: 0.4968
12704/25000 [==============>...............] - ETA: 40s - loss: 7.7113 - accuracy: 0.4971
12736/25000 [==============>...............] - ETA: 40s - loss: 7.7124 - accuracy: 0.4970
12768/25000 [==============>...............] - ETA: 39s - loss: 7.7135 - accuracy: 0.4969
12800/25000 [==============>...............] - ETA: 39s - loss: 7.7133 - accuracy: 0.4970
12832/25000 [==============>...............] - ETA: 39s - loss: 7.7108 - accuracy: 0.4971
12864/25000 [==============>...............] - ETA: 39s - loss: 7.7131 - accuracy: 0.4970
12896/25000 [==============>...............] - ETA: 39s - loss: 7.7118 - accuracy: 0.4971
12928/25000 [==============>...............] - ETA: 39s - loss: 7.7081 - accuracy: 0.4973
12960/25000 [==============>...............] - ETA: 39s - loss: 7.6986 - accuracy: 0.4979
12992/25000 [==============>...............] - ETA: 39s - loss: 7.7020 - accuracy: 0.4977
13024/25000 [==============>...............] - ETA: 39s - loss: 7.7008 - accuracy: 0.4978
13056/25000 [==============>...............] - ETA: 38s - loss: 7.7007 - accuracy: 0.4978
13088/25000 [==============>...............] - ETA: 38s - loss: 7.7006 - accuracy: 0.4978
13120/25000 [==============>...............] - ETA: 38s - loss: 7.6970 - accuracy: 0.4980
13152/25000 [==============>...............] - ETA: 38s - loss: 7.7063 - accuracy: 0.4974
13184/25000 [==============>...............] - ETA: 38s - loss: 7.7108 - accuracy: 0.4971
13216/25000 [==============>...............] - ETA: 38s - loss: 7.7119 - accuracy: 0.4970
13248/25000 [==============>...............] - ETA: 38s - loss: 7.7048 - accuracy: 0.4975
13280/25000 [==============>...............] - ETA: 38s - loss: 7.7059 - accuracy: 0.4974
13312/25000 [==============>...............] - ETA: 38s - loss: 7.7058 - accuracy: 0.4974
13344/25000 [===============>..............] - ETA: 38s - loss: 7.7034 - accuracy: 0.4976
13376/25000 [===============>..............] - ETA: 37s - loss: 7.7067 - accuracy: 0.4974
13408/25000 [===============>..............] - ETA: 37s - loss: 7.7066 - accuracy: 0.4974
13440/25000 [===============>..............] - ETA: 37s - loss: 7.7043 - accuracy: 0.4975
13472/25000 [===============>..............] - ETA: 37s - loss: 7.7053 - accuracy: 0.4975
13504/25000 [===============>..............] - ETA: 37s - loss: 7.7109 - accuracy: 0.4971
13536/25000 [===============>..............] - ETA: 37s - loss: 7.7119 - accuracy: 0.4970
13568/25000 [===============>..............] - ETA: 37s - loss: 7.7107 - accuracy: 0.4971
13600/25000 [===============>..............] - ETA: 37s - loss: 7.7140 - accuracy: 0.4969
13632/25000 [===============>..............] - ETA: 37s - loss: 7.7206 - accuracy: 0.4965
13664/25000 [===============>..............] - ETA: 36s - loss: 7.7194 - accuracy: 0.4966
13696/25000 [===============>..............] - ETA: 36s - loss: 7.7181 - accuracy: 0.4966
13728/25000 [===============>..............] - ETA: 36s - loss: 7.7146 - accuracy: 0.4969
13760/25000 [===============>..............] - ETA: 36s - loss: 7.7145 - accuracy: 0.4969
13792/25000 [===============>..............] - ETA: 36s - loss: 7.7200 - accuracy: 0.4965
13824/25000 [===============>..............] - ETA: 36s - loss: 7.7132 - accuracy: 0.4970
13856/25000 [===============>..............] - ETA: 36s - loss: 7.7120 - accuracy: 0.4970
13888/25000 [===============>..............] - ETA: 36s - loss: 7.7185 - accuracy: 0.4966
13920/25000 [===============>..............] - ETA: 36s - loss: 7.7184 - accuracy: 0.4966
13952/25000 [===============>..............] - ETA: 36s - loss: 7.7172 - accuracy: 0.4967
13984/25000 [===============>..............] - ETA: 35s - loss: 7.7214 - accuracy: 0.4964
14016/25000 [===============>..............] - ETA: 35s - loss: 7.7191 - accuracy: 0.4966
14048/25000 [===============>..............] - ETA: 35s - loss: 7.7179 - accuracy: 0.4967
14080/25000 [===============>..............] - ETA: 35s - loss: 7.7113 - accuracy: 0.4971
14112/25000 [===============>..............] - ETA: 35s - loss: 7.7112 - accuracy: 0.4971
14144/25000 [===============>..............] - ETA: 35s - loss: 7.7143 - accuracy: 0.4969
14176/25000 [================>.............] - ETA: 35s - loss: 7.7131 - accuracy: 0.4970
14208/25000 [================>.............] - ETA: 35s - loss: 7.7173 - accuracy: 0.4967
14240/25000 [================>.............] - ETA: 35s - loss: 7.7140 - accuracy: 0.4969
14272/25000 [================>.............] - ETA: 34s - loss: 7.7139 - accuracy: 0.4969
14304/25000 [================>.............] - ETA: 34s - loss: 7.7138 - accuracy: 0.4969
14336/25000 [================>.............] - ETA: 34s - loss: 7.7126 - accuracy: 0.4970
14368/25000 [================>.............] - ETA: 34s - loss: 7.7072 - accuracy: 0.4974
14400/25000 [================>.............] - ETA: 34s - loss: 7.7071 - accuracy: 0.4974
14432/25000 [================>.............] - ETA: 34s - loss: 7.7123 - accuracy: 0.4970
14464/25000 [================>.............] - ETA: 34s - loss: 7.7090 - accuracy: 0.4972
14496/25000 [================>.............] - ETA: 34s - loss: 7.7047 - accuracy: 0.4975
14528/25000 [================>.............] - ETA: 34s - loss: 7.7099 - accuracy: 0.4972
14560/25000 [================>.............] - ETA: 34s - loss: 7.7151 - accuracy: 0.4968
14592/25000 [================>.............] - ETA: 34s - loss: 7.7150 - accuracy: 0.4968
14624/25000 [================>.............] - ETA: 33s - loss: 7.7201 - accuracy: 0.4965
14656/25000 [================>.............] - ETA: 33s - loss: 7.7231 - accuracy: 0.4963
14688/25000 [================>.............] - ETA: 33s - loss: 7.7272 - accuracy: 0.4961
14720/25000 [================>.............] - ETA: 33s - loss: 7.7270 - accuracy: 0.4961
14752/25000 [================>.............] - ETA: 33s - loss: 7.7238 - accuracy: 0.4963
14784/25000 [================>.............] - ETA: 33s - loss: 7.7206 - accuracy: 0.4965
14816/25000 [================>.............] - ETA: 33s - loss: 7.7225 - accuracy: 0.4964
14848/25000 [================>.............] - ETA: 33s - loss: 7.7214 - accuracy: 0.4964
14880/25000 [================>.............] - ETA: 33s - loss: 7.7223 - accuracy: 0.4964
14912/25000 [================>.............] - ETA: 33s - loss: 7.7211 - accuracy: 0.4964
14944/25000 [================>.............] - ETA: 33s - loss: 7.7169 - accuracy: 0.4967
14976/25000 [================>.............] - ETA: 32s - loss: 7.7158 - accuracy: 0.4968
15008/25000 [=================>............] - ETA: 32s - loss: 7.7157 - accuracy: 0.4968
15040/25000 [=================>............] - ETA: 32s - loss: 7.7115 - accuracy: 0.4971
15072/25000 [=================>............] - ETA: 32s - loss: 7.7083 - accuracy: 0.4973
15104/25000 [=================>............] - ETA: 32s - loss: 7.7062 - accuracy: 0.4974
15136/25000 [=================>............] - ETA: 32s - loss: 7.7051 - accuracy: 0.4975
15168/25000 [=================>............] - ETA: 32s - loss: 7.7081 - accuracy: 0.4973
15200/25000 [=================>............] - ETA: 32s - loss: 7.7070 - accuracy: 0.4974
15232/25000 [=================>............] - ETA: 32s - loss: 7.7059 - accuracy: 0.4974
15264/25000 [=================>............] - ETA: 31s - loss: 7.7068 - accuracy: 0.4974
15296/25000 [=================>............] - ETA: 31s - loss: 7.7127 - accuracy: 0.4970
15328/25000 [=================>............] - ETA: 31s - loss: 7.7076 - accuracy: 0.4973
15360/25000 [=================>............] - ETA: 31s - loss: 7.7056 - accuracy: 0.4975
15392/25000 [=================>............] - ETA: 31s - loss: 7.7025 - accuracy: 0.4977
15424/25000 [=================>............] - ETA: 31s - loss: 7.7034 - accuracy: 0.4976
15456/25000 [=================>............] - ETA: 31s - loss: 7.7033 - accuracy: 0.4976
15488/25000 [=================>............] - ETA: 31s - loss: 7.7062 - accuracy: 0.4974
15520/25000 [=================>............] - ETA: 31s - loss: 7.7101 - accuracy: 0.4972
15552/25000 [=================>............] - ETA: 31s - loss: 7.7120 - accuracy: 0.4970
15584/25000 [=================>............] - ETA: 30s - loss: 7.7119 - accuracy: 0.4970
15616/25000 [=================>............] - ETA: 30s - loss: 7.7118 - accuracy: 0.4971
15648/25000 [=================>............] - ETA: 30s - loss: 7.7117 - accuracy: 0.4971
15680/25000 [=================>............] - ETA: 30s - loss: 7.7077 - accuracy: 0.4973
15712/25000 [=================>............] - ETA: 30s - loss: 7.7066 - accuracy: 0.4974
15744/25000 [=================>............] - ETA: 30s - loss: 7.7085 - accuracy: 0.4973
15776/25000 [=================>............] - ETA: 30s - loss: 7.7065 - accuracy: 0.4974
15808/25000 [=================>............] - ETA: 30s - loss: 7.7015 - accuracy: 0.4977
15840/25000 [==================>...........] - ETA: 30s - loss: 7.7005 - accuracy: 0.4978
15872/25000 [==================>...........] - ETA: 29s - loss: 7.7033 - accuracy: 0.4976
15904/25000 [==================>...........] - ETA: 29s - loss: 7.7042 - accuracy: 0.4975
15936/25000 [==================>...........] - ETA: 29s - loss: 7.7080 - accuracy: 0.4973
15968/25000 [==================>...........] - ETA: 29s - loss: 7.7089 - accuracy: 0.4972
16000/25000 [==================>...........] - ETA: 29s - loss: 7.7126 - accuracy: 0.4970
16032/25000 [==================>...........] - ETA: 29s - loss: 7.7087 - accuracy: 0.4973
16064/25000 [==================>...........] - ETA: 29s - loss: 7.7105 - accuracy: 0.4971
16096/25000 [==================>...........] - ETA: 29s - loss: 7.7114 - accuracy: 0.4971
16128/25000 [==================>...........] - ETA: 29s - loss: 7.7132 - accuracy: 0.4970
16160/25000 [==================>...........] - ETA: 28s - loss: 7.7150 - accuracy: 0.4968
16192/25000 [==================>...........] - ETA: 28s - loss: 7.7149 - accuracy: 0.4969
16224/25000 [==================>...........] - ETA: 28s - loss: 7.7158 - accuracy: 0.4968
16256/25000 [==================>...........] - ETA: 28s - loss: 7.7157 - accuracy: 0.4968
16288/25000 [==================>...........] - ETA: 28s - loss: 7.7165 - accuracy: 0.4967
16320/25000 [==================>...........] - ETA: 28s - loss: 7.7221 - accuracy: 0.4964
16352/25000 [==================>...........] - ETA: 28s - loss: 7.7257 - accuracy: 0.4961
16384/25000 [==================>...........] - ETA: 28s - loss: 7.7246 - accuracy: 0.4962
16416/25000 [==================>...........] - ETA: 28s - loss: 7.7264 - accuracy: 0.4961
16448/25000 [==================>...........] - ETA: 28s - loss: 7.7244 - accuracy: 0.4962
16480/25000 [==================>...........] - ETA: 27s - loss: 7.7234 - accuracy: 0.4963
16512/25000 [==================>...........] - ETA: 27s - loss: 7.7261 - accuracy: 0.4961
16544/25000 [==================>...........] - ETA: 27s - loss: 7.7241 - accuracy: 0.4963
16576/25000 [==================>...........] - ETA: 27s - loss: 7.7267 - accuracy: 0.4961
16608/25000 [==================>...........] - ETA: 27s - loss: 7.7303 - accuracy: 0.4958
16640/25000 [==================>...........] - ETA: 27s - loss: 7.7284 - accuracy: 0.4960
16672/25000 [===================>..........] - ETA: 27s - loss: 7.7301 - accuracy: 0.4959
16704/25000 [===================>..........] - ETA: 27s - loss: 7.7327 - accuracy: 0.4957
16736/25000 [===================>..........] - ETA: 27s - loss: 7.7317 - accuracy: 0.4958
16768/25000 [===================>..........] - ETA: 26s - loss: 7.7288 - accuracy: 0.4959
16800/25000 [===================>..........] - ETA: 26s - loss: 7.7269 - accuracy: 0.4961
16832/25000 [===================>..........] - ETA: 26s - loss: 7.7231 - accuracy: 0.4963
16864/25000 [===================>..........] - ETA: 26s - loss: 7.7266 - accuracy: 0.4961
16896/25000 [===================>..........] - ETA: 26s - loss: 7.7283 - accuracy: 0.4960
16928/25000 [===================>..........] - ETA: 26s - loss: 7.7273 - accuracy: 0.4960
16960/25000 [===================>..........] - ETA: 26s - loss: 7.7281 - accuracy: 0.4960
16992/25000 [===================>..........] - ETA: 26s - loss: 7.7307 - accuracy: 0.4958
17024/25000 [===================>..........] - ETA: 26s - loss: 7.7279 - accuracy: 0.4960
17056/25000 [===================>..........] - ETA: 26s - loss: 7.7260 - accuracy: 0.4961
17088/25000 [===================>..........] - ETA: 25s - loss: 7.7285 - accuracy: 0.4960
17120/25000 [===================>..........] - ETA: 25s - loss: 7.7284 - accuracy: 0.4960
17152/25000 [===================>..........] - ETA: 25s - loss: 7.7274 - accuracy: 0.4960
17184/25000 [===================>..........] - ETA: 25s - loss: 7.7255 - accuracy: 0.4962
17216/25000 [===================>..........] - ETA: 25s - loss: 7.7254 - accuracy: 0.4962
17248/25000 [===================>..........] - ETA: 25s - loss: 7.7262 - accuracy: 0.4961
17280/25000 [===================>..........] - ETA: 25s - loss: 7.7296 - accuracy: 0.4959
17312/25000 [===================>..........] - ETA: 25s - loss: 7.7330 - accuracy: 0.4957
17344/25000 [===================>..........] - ETA: 25s - loss: 7.7320 - accuracy: 0.4957
17376/25000 [===================>..........] - ETA: 24s - loss: 7.7319 - accuracy: 0.4957
17408/25000 [===================>..........] - ETA: 24s - loss: 7.7362 - accuracy: 0.4955
17440/25000 [===================>..........] - ETA: 24s - loss: 7.7396 - accuracy: 0.4952
17472/25000 [===================>..........] - ETA: 24s - loss: 7.7438 - accuracy: 0.4950
17504/25000 [====================>.........] - ETA: 24s - loss: 7.7420 - accuracy: 0.4951
17536/25000 [====================>.........] - ETA: 24s - loss: 7.7436 - accuracy: 0.4950
17568/25000 [====================>.........] - ETA: 24s - loss: 7.7478 - accuracy: 0.4947
17600/25000 [====================>.........] - ETA: 24s - loss: 7.7468 - accuracy: 0.4948
17632/25000 [====================>.........] - ETA: 24s - loss: 7.7492 - accuracy: 0.4946
17664/25000 [====================>.........] - ETA: 24s - loss: 7.7439 - accuracy: 0.4950
17696/25000 [====================>.........] - ETA: 23s - loss: 7.7394 - accuracy: 0.4953
17728/25000 [====================>.........] - ETA: 23s - loss: 7.7341 - accuracy: 0.4956
17760/25000 [====================>.........] - ETA: 23s - loss: 7.7305 - accuracy: 0.4958
17792/25000 [====================>.........] - ETA: 23s - loss: 7.7313 - accuracy: 0.4958
17824/25000 [====================>.........] - ETA: 23s - loss: 7.7294 - accuracy: 0.4959
17856/25000 [====================>.........] - ETA: 23s - loss: 7.7284 - accuracy: 0.4960
17888/25000 [====================>.........] - ETA: 23s - loss: 7.7283 - accuracy: 0.4960
17920/25000 [====================>.........] - ETA: 23s - loss: 7.7282 - accuracy: 0.4960
17952/25000 [====================>.........] - ETA: 23s - loss: 7.7281 - accuracy: 0.4960
17984/25000 [====================>.........] - ETA: 22s - loss: 7.7246 - accuracy: 0.4962
18016/25000 [====================>.........] - ETA: 22s - loss: 7.7228 - accuracy: 0.4963
18048/25000 [====================>.........] - ETA: 22s - loss: 7.7235 - accuracy: 0.4963
18080/25000 [====================>.........] - ETA: 22s - loss: 7.7226 - accuracy: 0.4963
18112/25000 [====================>.........] - ETA: 22s - loss: 7.7276 - accuracy: 0.4960
18144/25000 [====================>.........] - ETA: 22s - loss: 7.7275 - accuracy: 0.4960
18176/25000 [====================>.........] - ETA: 22s - loss: 7.7282 - accuracy: 0.4960
18208/25000 [====================>.........] - ETA: 22s - loss: 7.7298 - accuracy: 0.4959
18240/25000 [====================>.........] - ETA: 22s - loss: 7.7280 - accuracy: 0.4960
18272/25000 [====================>.........] - ETA: 22s - loss: 7.7287 - accuracy: 0.4960
18304/25000 [====================>.........] - ETA: 21s - loss: 7.7244 - accuracy: 0.4962
18336/25000 [=====================>........] - ETA: 21s - loss: 7.7260 - accuracy: 0.4961
18368/25000 [=====================>........] - ETA: 21s - loss: 7.7276 - accuracy: 0.4960
18400/25000 [=====================>........] - ETA: 21s - loss: 7.7275 - accuracy: 0.4960
18432/25000 [=====================>........] - ETA: 21s - loss: 7.7282 - accuracy: 0.4960
18464/25000 [=====================>........] - ETA: 21s - loss: 7.7248 - accuracy: 0.4962
18496/25000 [=====================>........] - ETA: 21s - loss: 7.7238 - accuracy: 0.4963
18528/25000 [=====================>........] - ETA: 21s - loss: 7.7245 - accuracy: 0.4962
18560/25000 [=====================>........] - ETA: 21s - loss: 7.7211 - accuracy: 0.4964
18592/25000 [=====================>........] - ETA: 20s - loss: 7.7202 - accuracy: 0.4965
18624/25000 [=====================>........] - ETA: 20s - loss: 7.7185 - accuracy: 0.4966
18656/25000 [=====================>........] - ETA: 20s - loss: 7.7200 - accuracy: 0.4965
18688/25000 [=====================>........] - ETA: 20s - loss: 7.7216 - accuracy: 0.4964
18720/25000 [=====================>........] - ETA: 20s - loss: 7.7207 - accuracy: 0.4965
18752/25000 [=====================>........] - ETA: 20s - loss: 7.7206 - accuracy: 0.4965
18784/25000 [=====================>........] - ETA: 20s - loss: 7.7180 - accuracy: 0.4966
18816/25000 [=====================>........] - ETA: 20s - loss: 7.7237 - accuracy: 0.4963
18848/25000 [=====================>........] - ETA: 20s - loss: 7.7284 - accuracy: 0.4960
18880/25000 [=====================>........] - ETA: 20s - loss: 7.7300 - accuracy: 0.4959
18912/25000 [=====================>........] - ETA: 19s - loss: 7.7299 - accuracy: 0.4959
18944/25000 [=====================>........] - ETA: 19s - loss: 7.7298 - accuracy: 0.4959
18976/25000 [=====================>........] - ETA: 19s - loss: 7.7264 - accuracy: 0.4961
19008/25000 [=====================>........] - ETA: 19s - loss: 7.7271 - accuracy: 0.4961
19040/25000 [=====================>........] - ETA: 19s - loss: 7.7343 - accuracy: 0.4956
19072/25000 [=====================>........] - ETA: 19s - loss: 7.7301 - accuracy: 0.4959
19104/25000 [=====================>........] - ETA: 19s - loss: 7.7364 - accuracy: 0.4954
19136/25000 [=====================>........] - ETA: 19s - loss: 7.7371 - accuracy: 0.4954
19168/25000 [======================>.......] - ETA: 19s - loss: 7.7354 - accuracy: 0.4955
19200/25000 [======================>.......] - ETA: 18s - loss: 7.7321 - accuracy: 0.4957
19232/25000 [======================>.......] - ETA: 18s - loss: 7.7328 - accuracy: 0.4957
19264/25000 [======================>.......] - ETA: 18s - loss: 7.7343 - accuracy: 0.4956
19296/25000 [======================>.......] - ETA: 18s - loss: 7.7342 - accuracy: 0.4956
19328/25000 [======================>.......] - ETA: 18s - loss: 7.7341 - accuracy: 0.4956
19360/25000 [======================>.......] - ETA: 18s - loss: 7.7339 - accuracy: 0.4956
19392/25000 [======================>.......] - ETA: 18s - loss: 7.7338 - accuracy: 0.4956
19424/25000 [======================>.......] - ETA: 18s - loss: 7.7337 - accuracy: 0.4956
19456/25000 [======================>.......] - ETA: 18s - loss: 7.7360 - accuracy: 0.4955
19488/25000 [======================>.......] - ETA: 18s - loss: 7.7359 - accuracy: 0.4955
19520/25000 [======================>.......] - ETA: 17s - loss: 7.7342 - accuracy: 0.4956
19552/25000 [======================>.......] - ETA: 17s - loss: 7.7325 - accuracy: 0.4957
19584/25000 [======================>.......] - ETA: 17s - loss: 7.7316 - accuracy: 0.4958
19616/25000 [======================>.......] - ETA: 17s - loss: 7.7346 - accuracy: 0.4956
19648/25000 [======================>.......] - ETA: 17s - loss: 7.7337 - accuracy: 0.4956
19680/25000 [======================>.......] - ETA: 17s - loss: 7.7305 - accuracy: 0.4958
19712/25000 [======================>.......] - ETA: 17s - loss: 7.7304 - accuracy: 0.4958
19744/25000 [======================>.......] - ETA: 17s - loss: 7.7295 - accuracy: 0.4959
19776/25000 [======================>.......] - ETA: 17s - loss: 7.7279 - accuracy: 0.4960
19808/25000 [======================>.......] - ETA: 17s - loss: 7.7255 - accuracy: 0.4962
19840/25000 [======================>.......] - ETA: 16s - loss: 7.7246 - accuracy: 0.4962
19872/25000 [======================>.......] - ETA: 16s - loss: 7.7291 - accuracy: 0.4959
19904/25000 [======================>.......] - ETA: 16s - loss: 7.7313 - accuracy: 0.4958
19936/25000 [======================>.......] - ETA: 16s - loss: 7.7274 - accuracy: 0.4960
19968/25000 [======================>.......] - ETA: 16s - loss: 7.7234 - accuracy: 0.4963
20000/25000 [=======================>......] - ETA: 16s - loss: 7.7241 - accuracy: 0.4963
20032/25000 [=======================>......] - ETA: 16s - loss: 7.7240 - accuracy: 0.4963
20064/25000 [=======================>......] - ETA: 16s - loss: 7.7194 - accuracy: 0.4966
20096/25000 [=======================>......] - ETA: 16s - loss: 7.7208 - accuracy: 0.4965
20128/25000 [=======================>......] - ETA: 15s - loss: 7.7215 - accuracy: 0.4964
20160/25000 [=======================>......] - ETA: 15s - loss: 7.7244 - accuracy: 0.4962
20192/25000 [=======================>......] - ETA: 15s - loss: 7.7251 - accuracy: 0.4962
20224/25000 [=======================>......] - ETA: 15s - loss: 7.7250 - accuracy: 0.4962
20256/25000 [=======================>......] - ETA: 15s - loss: 7.7257 - accuracy: 0.4961
20288/25000 [=======================>......] - ETA: 15s - loss: 7.7256 - accuracy: 0.4962
20320/25000 [=======================>......] - ETA: 15s - loss: 7.7194 - accuracy: 0.4966
20352/25000 [=======================>......] - ETA: 15s - loss: 7.7194 - accuracy: 0.4966
20384/25000 [=======================>......] - ETA: 15s - loss: 7.7200 - accuracy: 0.4965
20416/25000 [=======================>......] - ETA: 15s - loss: 7.7192 - accuracy: 0.4966
20448/25000 [=======================>......] - ETA: 14s - loss: 7.7154 - accuracy: 0.4968
20480/25000 [=======================>......] - ETA: 14s - loss: 7.7183 - accuracy: 0.4966
20512/25000 [=======================>......] - ETA: 14s - loss: 7.7204 - accuracy: 0.4965
20544/25000 [=======================>......] - ETA: 14s - loss: 7.7196 - accuracy: 0.4965
20576/25000 [=======================>......] - ETA: 14s - loss: 7.7218 - accuracy: 0.4964
20608/25000 [=======================>......] - ETA: 14s - loss: 7.7202 - accuracy: 0.4965
20640/25000 [=======================>......] - ETA: 14s - loss: 7.7201 - accuracy: 0.4965
20672/25000 [=======================>......] - ETA: 14s - loss: 7.7178 - accuracy: 0.4967
20704/25000 [=======================>......] - ETA: 14s - loss: 7.7177 - accuracy: 0.4967
20736/25000 [=======================>......] - ETA: 13s - loss: 7.7139 - accuracy: 0.4969
20768/25000 [=======================>......] - ETA: 13s - loss: 7.7131 - accuracy: 0.4970
20800/25000 [=======================>......] - ETA: 13s - loss: 7.7123 - accuracy: 0.4970
20832/25000 [=======================>......] - ETA: 13s - loss: 7.7137 - accuracy: 0.4969
20864/25000 [========================>.....] - ETA: 13s - loss: 7.7129 - accuracy: 0.4970
20896/25000 [========================>.....] - ETA: 13s - loss: 7.7165 - accuracy: 0.4967
20928/25000 [========================>.....] - ETA: 13s - loss: 7.7150 - accuracy: 0.4968
20960/25000 [========================>.....] - ETA: 13s - loss: 7.7171 - accuracy: 0.4967
20992/25000 [========================>.....] - ETA: 13s - loss: 7.7156 - accuracy: 0.4968
21024/25000 [========================>.....] - ETA: 13s - loss: 7.7148 - accuracy: 0.4969
21056/25000 [========================>.....] - ETA: 12s - loss: 7.7176 - accuracy: 0.4967
21088/25000 [========================>.....] - ETA: 12s - loss: 7.7182 - accuracy: 0.4966
21120/25000 [========================>.....] - ETA: 12s - loss: 7.7167 - accuracy: 0.4967
21152/25000 [========================>.....] - ETA: 12s - loss: 7.7152 - accuracy: 0.4968
21184/25000 [========================>.....] - ETA: 12s - loss: 7.7129 - accuracy: 0.4970
21216/25000 [========================>.....] - ETA: 12s - loss: 7.7150 - accuracy: 0.4968
21248/25000 [========================>.....] - ETA: 12s - loss: 7.7164 - accuracy: 0.4968
21280/25000 [========================>.....] - ETA: 12s - loss: 7.7149 - accuracy: 0.4969
21312/25000 [========================>.....] - ETA: 12s - loss: 7.7155 - accuracy: 0.4968
21344/25000 [========================>.....] - ETA: 11s - loss: 7.7133 - accuracy: 0.4970
21376/25000 [========================>.....] - ETA: 11s - loss: 7.7111 - accuracy: 0.4971
21408/25000 [========================>.....] - ETA: 11s - loss: 7.7117 - accuracy: 0.4971
21440/25000 [========================>.....] - ETA: 11s - loss: 7.7124 - accuracy: 0.4970
21472/25000 [========================>.....] - ETA: 11s - loss: 7.7102 - accuracy: 0.4972
21504/25000 [========================>.....] - ETA: 11s - loss: 7.7094 - accuracy: 0.4972
21536/25000 [========================>.....] - ETA: 11s - loss: 7.7072 - accuracy: 0.4974
21568/25000 [========================>.....] - ETA: 11s - loss: 7.7050 - accuracy: 0.4975
21600/25000 [========================>.....] - ETA: 11s - loss: 7.7057 - accuracy: 0.4975
21632/25000 [========================>.....] - ETA: 11s - loss: 7.7042 - accuracy: 0.4975
21664/25000 [========================>.....] - ETA: 10s - loss: 7.7041 - accuracy: 0.4976
21696/25000 [=========================>....] - ETA: 10s - loss: 7.7055 - accuracy: 0.4975
21728/25000 [=========================>....] - ETA: 10s - loss: 7.7068 - accuracy: 0.4974
21760/25000 [=========================>....] - ETA: 10s - loss: 7.7033 - accuracy: 0.4976
21792/25000 [=========================>....] - ETA: 10s - loss: 7.7025 - accuracy: 0.4977
21824/25000 [=========================>....] - ETA: 10s - loss: 7.7032 - accuracy: 0.4976
21856/25000 [=========================>....] - ETA: 10s - loss: 7.7066 - accuracy: 0.4974
21888/25000 [=========================>....] - ETA: 10s - loss: 7.7037 - accuracy: 0.4976
21920/25000 [=========================>....] - ETA: 10s - loss: 7.7016 - accuracy: 0.4977
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6994 - accuracy: 0.4979 
21984/25000 [=========================>....] - ETA: 9s - loss: 7.7008 - accuracy: 0.4978
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6980 - accuracy: 0.4980
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6958 - accuracy: 0.4981
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6930 - accuracy: 0.4983
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6902 - accuracy: 0.4985
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6895 - accuracy: 0.4985
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6901 - accuracy: 0.4985
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6915 - accuracy: 0.4984
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6901 - accuracy: 0.4985
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6893 - accuracy: 0.4985
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6900 - accuracy: 0.4985
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6879 - accuracy: 0.4986
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6838 - accuracy: 0.4989
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6844 - accuracy: 0.4988
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6844 - accuracy: 0.4988
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6823 - accuracy: 0.4990
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6857 - accuracy: 0.4988
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6843 - accuracy: 0.4988
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6843 - accuracy: 0.4988
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6849 - accuracy: 0.4988
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6788 - accuracy: 0.4992
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6781 - accuracy: 0.4993
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6781 - accuracy: 0.4993
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6781 - accuracy: 0.4993
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6774 - accuracy: 0.4993
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6767 - accuracy: 0.4993
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6760 - accuracy: 0.4994
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6767 - accuracy: 0.4993
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6773 - accuracy: 0.4993
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6780 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6760 - accuracy: 0.4994
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6766 - accuracy: 0.4993
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6786 - accuracy: 0.4992
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6792 - accuracy: 0.4992
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6812 - accuracy: 0.4990
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6779 - accuracy: 0.4993
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6785 - accuracy: 0.4992
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6805 - accuracy: 0.4991
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6818 - accuracy: 0.4990
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6844 - accuracy: 0.4988
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6824 - accuracy: 0.4990
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6850 - accuracy: 0.4988
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6804 - accuracy: 0.4991
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6810 - accuracy: 0.4991
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6791 - accuracy: 0.4992
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6790 - accuracy: 0.4992
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6810 - accuracy: 0.4991
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6797 - accuracy: 0.4991
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6770 - accuracy: 0.4993
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6751 - accuracy: 0.4994
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6757 - accuracy: 0.4994
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6711 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6698 - accuracy: 0.4998
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6685 - accuracy: 0.4999
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6685 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24192/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24256/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24320/25000 [============================>.] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24352/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24384/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24416/25000 [============================>.] - ETA: 1s - loss: 7.6742 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24480/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24608/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 98s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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


  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6cac74dfa76a4a815c2d5b8c43b2b7cd78357136', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6cac74dfa76a4a815c2d5b8c43b2b7cd78357136', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6cac74dfa76a4a815c2d5b8c43b2b7cd78357136

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6cac74dfa76a4a815c2d5b8c43b2b7cd78357136

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6cac74dfa76a4a815c2d5b8c43b2b7cd78357136

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files ################################
['ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//timeseries_m5_deepar.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py']





 ************************************************************************************************************************
############ Running Jupyter files ################################





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl.ipynb 

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
	Data preprocessing and feature engineering runtime = 0.2s ...
AutoGluon will gauge predictive performance using evaluation metric: accuracy
To change this, specify the eval_metric argument of fit()
AutoGluon will early stop models using evaluation metric: accuracy
Saving dataset/learner.pkl
Beginning hyperparameter tuning for Gradient Boosting Model...
Hyperparameter search space for Gradient Boosting Model: 
num_leaves:   Int: lower=26, upper=30
learning_rate:   Real: lower=0.005, upper=0.2
feature_fraction:   Real: lower=0.75, upper=1.0
min_data_in_leaf:   Int: lower=2, upper=30
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 283, in hyperparameter_tune
    directory=directory, lgb_model=self, **params_copy)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 69, in register_args
    self.update(**kwvars)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 79, in update
    hp = v.get_hp(name=k)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/space.py", line 451, in get_hp
    default_value=self._default)
  File "ConfigSpace/hyperparameters.pyx", line 773, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.__init__
  File "ConfigSpace/hyperparameters.pyx", line 843, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.check_default
Warning: Exception caused LightGBMClassifier to fail during hyperparameter tuning... Skipping this model.
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 283, in hyperparameter_tune
    directory=directory, lgb_model=self, **params_copy)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 69, in register_args
    self.update(**kwvars)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 79, in update
    hp = v.get_hp(name=k)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/space.py", line 451, in get_hp
    default_value=self._default)
  File "ConfigSpace/hyperparameters.pyx", line 773, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.__init__
  File "ConfigSpace/hyperparameters.pyx", line 843, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.check_default
ValueError: Illegal default value 36
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
 40%|████      | 2/5 [00:43<01:04, 21.66s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.48973322581234424, 'embedding_size_factor': 1.2112309861188437, 'layers.choice': 0, 'learning_rate': 0.00022093886271467536, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.1900097035181954e-11} and reward: 0.3644
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdfW\xca\x07(;\x02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3a3\xbe\x14\x84'X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?,\xf5za*/$X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xb8\x14TM2\x89Au." and reward: 0.3644
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdfW\xca\x07(;\x02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3a3\xbe\x14\x84'X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?,\xf5za*/$X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xb8\x14TM2\x89Au." and reward: 0.3644
 60%|██████    | 3/5 [01:27<00:56, 28.27s/it] 60%|██████    | 3/5 [01:27<00:58, 29.01s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.1616704042216712, 'embedding_size_factor': 0.9549500425018489, 'layers.choice': 2, 'learning_rate': 0.008092752164125437, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.1105172885001064e-09} and reward: 0.385
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc4\xb1\x9d\xa5n|\x9fX\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\x8e\xf3d;~\xfcX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x80\x92\xee\xcf\tR\x8cX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x13\x14\x1bO\xc9\x90\xbbu.' and reward: 0.385
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc4\xb1\x9d\xa5n|\x9fX\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\x8e\xf3d;~\xfcX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x80\x92\xee\xcf\tR\x8cX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x13\x14\x1bO\xc9\x90\xbbu.' and reward: 0.385
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 131.85385537147522
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.8s of the -14.07s of remaining time.
Ensemble size: 4
Ensemble weights: 
[0.75 0.25 0.  ]
	0.3946	 = Validation accuracy score
	0.98s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 135.08s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
test

  #### Module init   ############################################ 

  <module 'mlmodels.model_gluon.gluon_automl' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py'> 

  #### Loading params   ############################################## 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 456, in test_cli
    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py", line 109, in get_params
    return model_pars, data_pars, compute_pars, out_pars
UnboundLocalError: local variable 'model_pars' referenced before assignment





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras-textcnn.ipynb 

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
12845056/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-22 08:52:23.555922: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 08:52:23.559689: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-22 08:52:23.560013: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ca9770a390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 08:52:23.560030: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 3:39 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 2:15 - loss: 5.9895 - accuracy: 0.6094
   96/25000 [..............................] - ETA: 1:48 - loss: 6.0694 - accuracy: 0.6042
  128/25000 [..............................] - ETA: 1:36 - loss: 6.3489 - accuracy: 0.5859
  160/25000 [..............................] - ETA: 1:26 - loss: 6.3250 - accuracy: 0.5875
  192/25000 [..............................] - ETA: 1:21 - loss: 6.7083 - accuracy: 0.5625
  224/25000 [..............................] - ETA: 1:16 - loss: 6.8452 - accuracy: 0.5536
  256/25000 [..............................] - ETA: 1:13 - loss: 7.2474 - accuracy: 0.5273
  288/25000 [..............................] - ETA: 1:11 - loss: 7.4004 - accuracy: 0.5174
  320/25000 [..............................] - ETA: 1:11 - loss: 7.5229 - accuracy: 0.5094
  352/25000 [..............................] - ETA: 1:09 - loss: 7.4488 - accuracy: 0.5142
  384/25000 [..............................] - ETA: 1:08 - loss: 7.6267 - accuracy: 0.5026
  416/25000 [..............................] - ETA: 1:07 - loss: 7.5929 - accuracy: 0.5048
  448/25000 [..............................] - ETA: 1:06 - loss: 7.6666 - accuracy: 0.5000
  480/25000 [..............................] - ETA: 1:05 - loss: 7.8902 - accuracy: 0.4854
  512/25000 [..............................] - ETA: 1:05 - loss: 7.7864 - accuracy: 0.4922
  544/25000 [..............................] - ETA: 1:04 - loss: 7.8075 - accuracy: 0.4908
  576/25000 [..............................] - ETA: 1:03 - loss: 7.8530 - accuracy: 0.4878
  608/25000 [..............................] - ETA: 1:02 - loss: 7.7675 - accuracy: 0.4934
  640/25000 [..............................] - ETA: 1:02 - loss: 7.7864 - accuracy: 0.4922
  672/25000 [..............................] - ETA: 1:01 - loss: 7.8263 - accuracy: 0.4896
  704/25000 [..............................] - ETA: 1:00 - loss: 7.8626 - accuracy: 0.4872
  736/25000 [..............................] - ETA: 1:00 - loss: 7.7291 - accuracy: 0.4959
  768/25000 [..............................] - ETA: 59s - loss: 7.6467 - accuracy: 0.5013 
  800/25000 [..............................] - ETA: 59s - loss: 7.6091 - accuracy: 0.5038
  832/25000 [..............................] - ETA: 58s - loss: 7.6113 - accuracy: 0.5036
  864/25000 [>.............................] - ETA: 58s - loss: 7.6489 - accuracy: 0.5012
  896/25000 [>.............................] - ETA: 58s - loss: 7.7180 - accuracy: 0.4967
  928/25000 [>.............................] - ETA: 58s - loss: 7.6997 - accuracy: 0.4978
  960/25000 [>.............................] - ETA: 57s - loss: 7.7145 - accuracy: 0.4969
  992/25000 [>.............................] - ETA: 57s - loss: 7.7130 - accuracy: 0.4970
 1024/25000 [>.............................] - ETA: 57s - loss: 7.7115 - accuracy: 0.4971
 1056/25000 [>.............................] - ETA: 57s - loss: 7.6811 - accuracy: 0.4991
 1088/25000 [>.............................] - ETA: 56s - loss: 7.6948 - accuracy: 0.4982
 1120/25000 [>.............................] - ETA: 56s - loss: 7.6803 - accuracy: 0.4991
 1152/25000 [>.............................] - ETA: 56s - loss: 7.6533 - accuracy: 0.5009
 1184/25000 [>.............................] - ETA: 55s - loss: 7.6925 - accuracy: 0.4983
 1216/25000 [>.............................] - ETA: 55s - loss: 7.6792 - accuracy: 0.4992
 1248/25000 [>.............................] - ETA: 55s - loss: 7.7035 - accuracy: 0.4976
 1280/25000 [>.............................] - ETA: 55s - loss: 7.7385 - accuracy: 0.4953
 1312/25000 [>.............................] - ETA: 55s - loss: 7.7952 - accuracy: 0.4916
 1344/25000 [>.............................] - ETA: 55s - loss: 7.7693 - accuracy: 0.4933
 1376/25000 [>.............................] - ETA: 54s - loss: 7.7781 - accuracy: 0.4927
 1408/25000 [>.............................] - ETA: 54s - loss: 7.8300 - accuracy: 0.4893
 1440/25000 [>.............................] - ETA: 54s - loss: 7.8263 - accuracy: 0.4896
 1472/25000 [>.............................] - ETA: 54s - loss: 7.7708 - accuracy: 0.4932
 1504/25000 [>.............................] - ETA: 54s - loss: 7.7686 - accuracy: 0.4934
 1536/25000 [>.............................] - ETA: 54s - loss: 7.8064 - accuracy: 0.4909
 1568/25000 [>.............................] - ETA: 54s - loss: 7.8329 - accuracy: 0.4892
 1600/25000 [>.............................] - ETA: 53s - loss: 7.8295 - accuracy: 0.4894
 1632/25000 [>.............................] - ETA: 53s - loss: 7.8545 - accuracy: 0.4877
 1664/25000 [>.............................] - ETA: 53s - loss: 7.8878 - accuracy: 0.4856
 1696/25000 [=>............................] - ETA: 53s - loss: 7.8655 - accuracy: 0.4870
 1728/25000 [=>............................] - ETA: 53s - loss: 7.8441 - accuracy: 0.4884
 1760/25000 [=>............................] - ETA: 52s - loss: 7.8583 - accuracy: 0.4875
 1792/25000 [=>............................] - ETA: 52s - loss: 7.8549 - accuracy: 0.4877
 1824/25000 [=>............................] - ETA: 52s - loss: 7.8516 - accuracy: 0.4879
 1856/25000 [=>............................] - ETA: 52s - loss: 7.7823 - accuracy: 0.4925
 1888/25000 [=>............................] - ETA: 52s - loss: 7.7722 - accuracy: 0.4931
 1920/25000 [=>............................] - ETA: 52s - loss: 7.7864 - accuracy: 0.4922
 1952/25000 [=>............................] - ETA: 52s - loss: 7.8002 - accuracy: 0.4913
 1984/25000 [=>............................] - ETA: 51s - loss: 7.8212 - accuracy: 0.4899
 2016/25000 [=>............................] - ETA: 51s - loss: 7.7579 - accuracy: 0.4940
 2048/25000 [=>............................] - ETA: 51s - loss: 7.7789 - accuracy: 0.4927
 2080/25000 [=>............................] - ETA: 51s - loss: 7.7625 - accuracy: 0.4938
 2112/25000 [=>............................] - ETA: 51s - loss: 7.7683 - accuracy: 0.4934
 2144/25000 [=>............................] - ETA: 51s - loss: 7.7810 - accuracy: 0.4925
 2176/25000 [=>............................] - ETA: 51s - loss: 7.7582 - accuracy: 0.4940
 2208/25000 [=>............................] - ETA: 51s - loss: 7.7430 - accuracy: 0.4950
 2240/25000 [=>............................] - ETA: 51s - loss: 7.7625 - accuracy: 0.4938
 2272/25000 [=>............................] - ETA: 51s - loss: 7.7544 - accuracy: 0.4943
 2304/25000 [=>............................] - ETA: 51s - loss: 7.7332 - accuracy: 0.4957
 2336/25000 [=>............................] - ETA: 50s - loss: 7.7126 - accuracy: 0.4970
 2368/25000 [=>............................] - ETA: 50s - loss: 7.7184 - accuracy: 0.4966
 2400/25000 [=>............................] - ETA: 50s - loss: 7.6922 - accuracy: 0.4983
 2432/25000 [=>............................] - ETA: 50s - loss: 7.6792 - accuracy: 0.4992
 2464/25000 [=>............................] - ETA: 50s - loss: 7.6791 - accuracy: 0.4992
 2496/25000 [=>............................] - ETA: 50s - loss: 7.6850 - accuracy: 0.4988
 2528/25000 [==>...........................] - ETA: 50s - loss: 7.7151 - accuracy: 0.4968
 2560/25000 [==>...........................] - ETA: 50s - loss: 7.7026 - accuracy: 0.4977
 2592/25000 [==>...........................] - ETA: 50s - loss: 7.7021 - accuracy: 0.4977
 2624/25000 [==>...........................] - ETA: 49s - loss: 7.7017 - accuracy: 0.4977
 2656/25000 [==>...........................] - ETA: 49s - loss: 7.7186 - accuracy: 0.4966
 2688/25000 [==>...........................] - ETA: 49s - loss: 7.7065 - accuracy: 0.4974
 2720/25000 [==>...........................] - ETA: 49s - loss: 7.6723 - accuracy: 0.4996
 2752/25000 [==>...........................] - ETA: 49s - loss: 7.6778 - accuracy: 0.4993
 2784/25000 [==>...........................] - ETA: 49s - loss: 7.6887 - accuracy: 0.4986
 2816/25000 [==>...........................] - ETA: 49s - loss: 7.7047 - accuracy: 0.4975
 2848/25000 [==>...........................] - ETA: 49s - loss: 7.7258 - accuracy: 0.4961
 2880/25000 [==>...........................] - ETA: 49s - loss: 7.7092 - accuracy: 0.4972
 2912/25000 [==>...........................] - ETA: 48s - loss: 7.7140 - accuracy: 0.4969
 2944/25000 [==>...........................] - ETA: 48s - loss: 7.7187 - accuracy: 0.4966
 2976/25000 [==>...........................] - ETA: 48s - loss: 7.6821 - accuracy: 0.4990
 3008/25000 [==>...........................] - ETA: 48s - loss: 7.6564 - accuracy: 0.5007
 3040/25000 [==>...........................] - ETA: 48s - loss: 7.6515 - accuracy: 0.5010
 3072/25000 [==>...........................] - ETA: 48s - loss: 7.6417 - accuracy: 0.5016
 3104/25000 [==>...........................] - ETA: 48s - loss: 7.6419 - accuracy: 0.5016
 3136/25000 [==>...........................] - ETA: 48s - loss: 7.6373 - accuracy: 0.5019
 3168/25000 [==>...........................] - ETA: 48s - loss: 7.6279 - accuracy: 0.5025
 3200/25000 [==>...........................] - ETA: 48s - loss: 7.6187 - accuracy: 0.5031
 3232/25000 [==>...........................] - ETA: 48s - loss: 7.6334 - accuracy: 0.5022
 3264/25000 [==>...........................] - ETA: 47s - loss: 7.6290 - accuracy: 0.5025
 3296/25000 [==>...........................] - ETA: 47s - loss: 7.6341 - accuracy: 0.5021
 3328/25000 [==>...........................] - ETA: 47s - loss: 7.6252 - accuracy: 0.5027
 3360/25000 [===>..........................] - ETA: 47s - loss: 7.6255 - accuracy: 0.5027
 3392/25000 [===>..........................] - ETA: 47s - loss: 7.6305 - accuracy: 0.5024
 3424/25000 [===>..........................] - ETA: 47s - loss: 7.6263 - accuracy: 0.5026
 3456/25000 [===>..........................] - ETA: 47s - loss: 7.6444 - accuracy: 0.5014
 3488/25000 [===>..........................] - ETA: 47s - loss: 7.6402 - accuracy: 0.5017
 3520/25000 [===>..........................] - ETA: 47s - loss: 7.6405 - accuracy: 0.5017
 3552/25000 [===>..........................] - ETA: 47s - loss: 7.6278 - accuracy: 0.5025
 3584/25000 [===>..........................] - ETA: 47s - loss: 7.5982 - accuracy: 0.5045
 3616/25000 [===>..........................] - ETA: 47s - loss: 7.5818 - accuracy: 0.5055
 3648/25000 [===>..........................] - ETA: 47s - loss: 7.5952 - accuracy: 0.5047
 3680/25000 [===>..........................] - ETA: 47s - loss: 7.5750 - accuracy: 0.5060
 3712/25000 [===>..........................] - ETA: 46s - loss: 7.6005 - accuracy: 0.5043
 3744/25000 [===>..........................] - ETA: 46s - loss: 7.6134 - accuracy: 0.5035
 3776/25000 [===>..........................] - ETA: 46s - loss: 7.5976 - accuracy: 0.5045
 3808/25000 [===>..........................] - ETA: 46s - loss: 7.6022 - accuracy: 0.5042
 3840/25000 [===>..........................] - ETA: 46s - loss: 7.5868 - accuracy: 0.5052
 3872/25000 [===>..........................] - ETA: 46s - loss: 7.5835 - accuracy: 0.5054
 3904/25000 [===>..........................] - ETA: 46s - loss: 7.5763 - accuracy: 0.5059
 3936/25000 [===>..........................] - ETA: 46s - loss: 7.5848 - accuracy: 0.5053
 3968/25000 [===>..........................] - ETA: 46s - loss: 7.5893 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 46s - loss: 7.6091 - accuracy: 0.5038
 4032/25000 [===>..........................] - ETA: 46s - loss: 7.6020 - accuracy: 0.5042
 4064/25000 [===>..........................] - ETA: 46s - loss: 7.6213 - accuracy: 0.5030
 4096/25000 [===>..........................] - ETA: 45s - loss: 7.6105 - accuracy: 0.5037
 4128/25000 [===>..........................] - ETA: 45s - loss: 7.6220 - accuracy: 0.5029
 4160/25000 [===>..........................] - ETA: 45s - loss: 7.6187 - accuracy: 0.5031
 4192/25000 [====>.........................] - ETA: 45s - loss: 7.6081 - accuracy: 0.5038
 4224/25000 [====>.........................] - ETA: 45s - loss: 7.5976 - accuracy: 0.5045
 4256/25000 [====>.........................] - ETA: 45s - loss: 7.5946 - accuracy: 0.5047
 4288/25000 [====>.........................] - ETA: 45s - loss: 7.6058 - accuracy: 0.5040
 4320/25000 [====>.........................] - ETA: 45s - loss: 7.6098 - accuracy: 0.5037
 4352/25000 [====>.........................] - ETA: 45s - loss: 7.6243 - accuracy: 0.5028
 4384/25000 [====>.........................] - ETA: 45s - loss: 7.6212 - accuracy: 0.5030
 4416/25000 [====>.........................] - ETA: 45s - loss: 7.6041 - accuracy: 0.5041
 4448/25000 [====>.........................] - ETA: 45s - loss: 7.6011 - accuracy: 0.5043
 4480/25000 [====>.........................] - ETA: 44s - loss: 7.6016 - accuracy: 0.5042
 4512/25000 [====>.........................] - ETA: 44s - loss: 7.5919 - accuracy: 0.5049
 4544/25000 [====>.........................] - ETA: 44s - loss: 7.5856 - accuracy: 0.5053
 4576/25000 [====>.........................] - ETA: 44s - loss: 7.6030 - accuracy: 0.5042
 4608/25000 [====>.........................] - ETA: 44s - loss: 7.6001 - accuracy: 0.5043
 4640/25000 [====>.........................] - ETA: 44s - loss: 7.5939 - accuracy: 0.5047
 4672/25000 [====>.........................] - ETA: 44s - loss: 7.5911 - accuracy: 0.5049
 4704/25000 [====>.........................] - ETA: 44s - loss: 7.5916 - accuracy: 0.5049
 4736/25000 [====>.........................] - ETA: 44s - loss: 7.5792 - accuracy: 0.5057
 4768/25000 [====>.........................] - ETA: 44s - loss: 7.5798 - accuracy: 0.5057
 4800/25000 [====>.........................] - ETA: 44s - loss: 7.5868 - accuracy: 0.5052
 4832/25000 [====>.........................] - ETA: 44s - loss: 7.5809 - accuracy: 0.5056
 4864/25000 [====>.........................] - ETA: 43s - loss: 7.5752 - accuracy: 0.5060
 4896/25000 [====>.........................] - ETA: 43s - loss: 7.5915 - accuracy: 0.5049
 4928/25000 [====>.........................] - ETA: 43s - loss: 7.5951 - accuracy: 0.5047
 4960/25000 [====>.........................] - ETA: 43s - loss: 7.6017 - accuracy: 0.5042
 4992/25000 [====>.........................] - ETA: 43s - loss: 7.6052 - accuracy: 0.5040
 5024/25000 [=====>........................] - ETA: 43s - loss: 7.5995 - accuracy: 0.5044
 5056/25000 [=====>........................] - ETA: 43s - loss: 7.5999 - accuracy: 0.5044
 5088/25000 [=====>........................] - ETA: 43s - loss: 7.5973 - accuracy: 0.5045
 5120/25000 [=====>........................] - ETA: 43s - loss: 7.5947 - accuracy: 0.5047
 5152/25000 [=====>........................] - ETA: 43s - loss: 7.5833 - accuracy: 0.5054
 5184/25000 [=====>........................] - ETA: 43s - loss: 7.5868 - accuracy: 0.5052
 5216/25000 [=====>........................] - ETA: 43s - loss: 7.5961 - accuracy: 0.5046
 5248/25000 [=====>........................] - ETA: 43s - loss: 7.5994 - accuracy: 0.5044
 5280/25000 [=====>........................] - ETA: 42s - loss: 7.5824 - accuracy: 0.5055
 5312/25000 [=====>........................] - ETA: 42s - loss: 7.5800 - accuracy: 0.5056
 5344/25000 [=====>........................] - ETA: 42s - loss: 7.5863 - accuracy: 0.5052
 5376/25000 [=====>........................] - ETA: 42s - loss: 7.5811 - accuracy: 0.5056
 5408/25000 [=====>........................] - ETA: 42s - loss: 7.5787 - accuracy: 0.5057
 5440/25000 [=====>........................] - ETA: 42s - loss: 7.5877 - accuracy: 0.5051
 5472/25000 [=====>........................] - ETA: 42s - loss: 7.5854 - accuracy: 0.5053
 5504/25000 [=====>........................] - ETA: 42s - loss: 7.5775 - accuracy: 0.5058
 5536/25000 [=====>........................] - ETA: 42s - loss: 7.5863 - accuracy: 0.5052
 5568/25000 [=====>........................] - ETA: 42s - loss: 7.5785 - accuracy: 0.5057
 5600/25000 [=====>........................] - ETA: 42s - loss: 7.5735 - accuracy: 0.5061
 5632/25000 [=====>........................] - ETA: 42s - loss: 7.5822 - accuracy: 0.5055
 5664/25000 [=====>........................] - ETA: 42s - loss: 7.5800 - accuracy: 0.5056
 5696/25000 [=====>........................] - ETA: 42s - loss: 7.5832 - accuracy: 0.5054
 5728/25000 [=====>........................] - ETA: 41s - loss: 7.5836 - accuracy: 0.5054
 5760/25000 [=====>........................] - ETA: 41s - loss: 7.5814 - accuracy: 0.5056
 5792/25000 [=====>........................] - ETA: 41s - loss: 7.5872 - accuracy: 0.5052
 5824/25000 [=====>........................] - ETA: 41s - loss: 7.5718 - accuracy: 0.5062
 5856/25000 [======>.......................] - ETA: 41s - loss: 7.5724 - accuracy: 0.5061
 5888/25000 [======>.......................] - ETA: 41s - loss: 7.5703 - accuracy: 0.5063
 5920/25000 [======>.......................] - ETA: 41s - loss: 7.5578 - accuracy: 0.5071
 5952/25000 [======>.......................] - ETA: 41s - loss: 7.5610 - accuracy: 0.5069
 5984/25000 [======>.......................] - ETA: 41s - loss: 7.5564 - accuracy: 0.5072
 6016/25000 [======>.......................] - ETA: 41s - loss: 7.5468 - accuracy: 0.5078
 6048/25000 [======>.......................] - ETA: 41s - loss: 7.5652 - accuracy: 0.5066
 6080/25000 [======>.......................] - ETA: 41s - loss: 7.5683 - accuracy: 0.5064
 6112/25000 [======>.......................] - ETA: 41s - loss: 7.5638 - accuracy: 0.5067
 6144/25000 [======>.......................] - ETA: 41s - loss: 7.5618 - accuracy: 0.5068
 6176/25000 [======>.......................] - ETA: 40s - loss: 7.5698 - accuracy: 0.5063
 6208/25000 [======>.......................] - ETA: 40s - loss: 7.5752 - accuracy: 0.5060
 6240/25000 [======>.......................] - ETA: 40s - loss: 7.5683 - accuracy: 0.5064
 6272/25000 [======>.......................] - ETA: 40s - loss: 7.5591 - accuracy: 0.5070
 6304/25000 [======>.......................] - ETA: 40s - loss: 7.5547 - accuracy: 0.5073
 6336/25000 [======>.......................] - ETA: 40s - loss: 7.5529 - accuracy: 0.5074
 6368/25000 [======>.......................] - ETA: 40s - loss: 7.5510 - accuracy: 0.5075
 6400/25000 [======>.......................] - ETA: 40s - loss: 7.5516 - accuracy: 0.5075
 6432/25000 [======>.......................] - ETA: 40s - loss: 7.5689 - accuracy: 0.5064
 6464/25000 [======>.......................] - ETA: 40s - loss: 7.5741 - accuracy: 0.5060
 6496/25000 [======>.......................] - ETA: 40s - loss: 7.5698 - accuracy: 0.5063
 6528/25000 [======>.......................] - ETA: 40s - loss: 7.5703 - accuracy: 0.5063
 6560/25000 [======>.......................] - ETA: 39s - loss: 7.5778 - accuracy: 0.5058
 6592/25000 [======>.......................] - ETA: 39s - loss: 7.5713 - accuracy: 0.5062
 6624/25000 [======>.......................] - ETA: 39s - loss: 7.5717 - accuracy: 0.5062
 6656/25000 [======>.......................] - ETA: 39s - loss: 7.5837 - accuracy: 0.5054
 6688/25000 [=======>......................] - ETA: 39s - loss: 7.5841 - accuracy: 0.5054
 6720/25000 [=======>......................] - ETA: 39s - loss: 7.5845 - accuracy: 0.5054
 6752/25000 [=======>......................] - ETA: 39s - loss: 7.5849 - accuracy: 0.5053
 6784/25000 [=======>......................] - ETA: 39s - loss: 7.5966 - accuracy: 0.5046
 6816/25000 [=======>......................] - ETA: 39s - loss: 7.5946 - accuracy: 0.5047
 6848/25000 [=======>......................] - ETA: 39s - loss: 7.6017 - accuracy: 0.5042
 6880/25000 [=======>......................] - ETA: 39s - loss: 7.6042 - accuracy: 0.5041
 6912/25000 [=======>......................] - ETA: 39s - loss: 7.5956 - accuracy: 0.5046
 6944/25000 [=======>......................] - ETA: 39s - loss: 7.5915 - accuracy: 0.5049
 6976/25000 [=======>......................] - ETA: 39s - loss: 7.5831 - accuracy: 0.5054
 7008/25000 [=======>......................] - ETA: 39s - loss: 7.5791 - accuracy: 0.5057
 7040/25000 [=======>......................] - ETA: 38s - loss: 7.5795 - accuracy: 0.5057
 7072/25000 [=======>......................] - ETA: 38s - loss: 7.5777 - accuracy: 0.5058
 7104/25000 [=======>......................] - ETA: 38s - loss: 7.5717 - accuracy: 0.5062
 7136/25000 [=======>......................] - ETA: 38s - loss: 7.5721 - accuracy: 0.5062
 7168/25000 [=======>......................] - ETA: 38s - loss: 7.5725 - accuracy: 0.5061
 7200/25000 [=======>......................] - ETA: 38s - loss: 7.5750 - accuracy: 0.5060
 7232/25000 [=======>......................] - ETA: 38s - loss: 7.5755 - accuracy: 0.5059
 7264/25000 [=======>......................] - ETA: 38s - loss: 7.5780 - accuracy: 0.5058
 7296/25000 [=======>......................] - ETA: 38s - loss: 7.5784 - accuracy: 0.5058
 7328/25000 [=======>......................] - ETA: 38s - loss: 7.5725 - accuracy: 0.5061
 7360/25000 [=======>......................] - ETA: 38s - loss: 7.5791 - accuracy: 0.5057
 7392/25000 [=======>......................] - ETA: 38s - loss: 7.5795 - accuracy: 0.5057
 7424/25000 [=======>......................] - ETA: 38s - loss: 7.5985 - accuracy: 0.5044
 7456/25000 [=======>......................] - ETA: 38s - loss: 7.5946 - accuracy: 0.5047
 7488/25000 [=======>......................] - ETA: 37s - loss: 7.6052 - accuracy: 0.5040
 7520/25000 [========>.....................] - ETA: 37s - loss: 7.6034 - accuracy: 0.5041
 7552/25000 [========>.....................] - ETA: 37s - loss: 7.6016 - accuracy: 0.5042
 7584/25000 [========>.....................] - ETA: 37s - loss: 7.5938 - accuracy: 0.5047
 7616/25000 [========>.....................] - ETA: 37s - loss: 7.5941 - accuracy: 0.5047
 7648/25000 [========>.....................] - ETA: 37s - loss: 7.5964 - accuracy: 0.5046
 7680/25000 [========>.....................] - ETA: 37s - loss: 7.5967 - accuracy: 0.5046
 7712/25000 [========>.....................] - ETA: 37s - loss: 7.5970 - accuracy: 0.5045
 7744/25000 [========>.....................] - ETA: 37s - loss: 7.5914 - accuracy: 0.5049
 7776/25000 [========>.....................] - ETA: 37s - loss: 7.5937 - accuracy: 0.5048
 7808/25000 [========>.....................] - ETA: 37s - loss: 7.5920 - accuracy: 0.5049
 7840/25000 [========>.....................] - ETA: 37s - loss: 7.5864 - accuracy: 0.5052
 7872/25000 [========>.....................] - ETA: 37s - loss: 7.5907 - accuracy: 0.5050
 7904/25000 [========>.....................] - ETA: 37s - loss: 7.5987 - accuracy: 0.5044
 7936/25000 [========>.....................] - ETA: 37s - loss: 7.5951 - accuracy: 0.5047
 7968/25000 [========>.....................] - ETA: 36s - loss: 7.6012 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 36s - loss: 7.6015 - accuracy: 0.5042
 8032/25000 [========>.....................] - ETA: 36s - loss: 7.5979 - accuracy: 0.5045
 8064/25000 [========>.....................] - ETA: 36s - loss: 7.5925 - accuracy: 0.5048
 8096/25000 [========>.....................] - ETA: 36s - loss: 7.5852 - accuracy: 0.5053
 8128/25000 [========>.....................] - ETA: 36s - loss: 7.5761 - accuracy: 0.5059
 8160/25000 [========>.....................] - ETA: 36s - loss: 7.5783 - accuracy: 0.5058
 8192/25000 [========>.....................] - ETA: 36s - loss: 7.5824 - accuracy: 0.5055
 8224/25000 [========>.....................] - ETA: 36s - loss: 7.5827 - accuracy: 0.5055
 8256/25000 [========>.....................] - ETA: 36s - loss: 7.5886 - accuracy: 0.5051
 8288/25000 [========>.....................] - ETA: 36s - loss: 7.5889 - accuracy: 0.5051
 8320/25000 [========>.....................] - ETA: 36s - loss: 7.5892 - accuracy: 0.5050
 8352/25000 [=========>....................] - ETA: 36s - loss: 7.5895 - accuracy: 0.5050
 8384/25000 [=========>....................] - ETA: 36s - loss: 7.5935 - accuracy: 0.5048
 8416/25000 [=========>....................] - ETA: 36s - loss: 7.5956 - accuracy: 0.5046
 8448/25000 [=========>....................] - ETA: 35s - loss: 7.5995 - accuracy: 0.5044
 8480/25000 [=========>....................] - ETA: 35s - loss: 7.5907 - accuracy: 0.5050
 8512/25000 [=========>....................] - ETA: 35s - loss: 7.5928 - accuracy: 0.5048
 8544/25000 [=========>....................] - ETA: 35s - loss: 7.5912 - accuracy: 0.5049
 8576/25000 [=========>....................] - ETA: 35s - loss: 7.5933 - accuracy: 0.5048
 8608/25000 [=========>....................] - ETA: 35s - loss: 7.5936 - accuracy: 0.5048
 8640/25000 [=========>....................] - ETA: 35s - loss: 7.5956 - accuracy: 0.5046
 8672/25000 [=========>....................] - ETA: 35s - loss: 7.5924 - accuracy: 0.5048
 8704/25000 [=========>....................] - ETA: 35s - loss: 7.5891 - accuracy: 0.5051
 8736/25000 [=========>....................] - ETA: 35s - loss: 7.5824 - accuracy: 0.5055
 8768/25000 [=========>....................] - ETA: 35s - loss: 7.5792 - accuracy: 0.5057
 8800/25000 [=========>....................] - ETA: 35s - loss: 7.5760 - accuracy: 0.5059
 8832/25000 [=========>....................] - ETA: 35s - loss: 7.5833 - accuracy: 0.5054
 8864/25000 [=========>....................] - ETA: 35s - loss: 7.5905 - accuracy: 0.5050
 8896/25000 [=========>....................] - ETA: 34s - loss: 7.5873 - accuracy: 0.5052
 8928/25000 [=========>....................] - ETA: 34s - loss: 7.5876 - accuracy: 0.5052
 8960/25000 [=========>....................] - ETA: 34s - loss: 7.5828 - accuracy: 0.5055
 8992/25000 [=========>....................] - ETA: 34s - loss: 7.5848 - accuracy: 0.5053
 9024/25000 [=========>....................] - ETA: 34s - loss: 7.5902 - accuracy: 0.5050
 9056/25000 [=========>....................] - ETA: 34s - loss: 7.5887 - accuracy: 0.5051
 9088/25000 [=========>....................] - ETA: 34s - loss: 7.5823 - accuracy: 0.5055
 9120/25000 [=========>....................] - ETA: 34s - loss: 7.5926 - accuracy: 0.5048
 9152/25000 [=========>....................] - ETA: 34s - loss: 7.5845 - accuracy: 0.5054
 9184/25000 [==========>...................] - ETA: 34s - loss: 7.5831 - accuracy: 0.5054
 9216/25000 [==========>...................] - ETA: 34s - loss: 7.5768 - accuracy: 0.5059
 9248/25000 [==========>...................] - ETA: 34s - loss: 7.5721 - accuracy: 0.5062
 9280/25000 [==========>...................] - ETA: 34s - loss: 7.5757 - accuracy: 0.5059
 9312/25000 [==========>...................] - ETA: 34s - loss: 7.5761 - accuracy: 0.5059
 9344/25000 [==========>...................] - ETA: 33s - loss: 7.5780 - accuracy: 0.5058
 9376/25000 [==========>...................] - ETA: 33s - loss: 7.5734 - accuracy: 0.5061
 9408/25000 [==========>...................] - ETA: 33s - loss: 7.5639 - accuracy: 0.5067
 9440/25000 [==========>...................] - ETA: 33s - loss: 7.5643 - accuracy: 0.5067
 9472/25000 [==========>...................] - ETA: 33s - loss: 7.5549 - accuracy: 0.5073
 9504/25000 [==========>...................] - ETA: 33s - loss: 7.5537 - accuracy: 0.5074
 9536/25000 [==========>...................] - ETA: 33s - loss: 7.5525 - accuracy: 0.5074
 9568/25000 [==========>...................] - ETA: 33s - loss: 7.5528 - accuracy: 0.5074
 9600/25000 [==========>...................] - ETA: 33s - loss: 7.5580 - accuracy: 0.5071
 9632/25000 [==========>...................] - ETA: 33s - loss: 7.5600 - accuracy: 0.5070
 9664/25000 [==========>...................] - ETA: 33s - loss: 7.5603 - accuracy: 0.5069
 9696/25000 [==========>...................] - ETA: 33s - loss: 7.5607 - accuracy: 0.5069
 9728/25000 [==========>...................] - ETA: 33s - loss: 7.5547 - accuracy: 0.5073
 9760/25000 [==========>...................] - ETA: 32s - loss: 7.5614 - accuracy: 0.5069
 9792/25000 [==========>...................] - ETA: 32s - loss: 7.5601 - accuracy: 0.5069
 9824/25000 [==========>...................] - ETA: 32s - loss: 7.5574 - accuracy: 0.5071
 9856/25000 [==========>...................] - ETA: 32s - loss: 7.5593 - accuracy: 0.5070
 9888/25000 [==========>...................] - ETA: 32s - loss: 7.5565 - accuracy: 0.5072
 9920/25000 [==========>...................] - ETA: 32s - loss: 7.5631 - accuracy: 0.5068
 9952/25000 [==========>...................] - ETA: 32s - loss: 7.5634 - accuracy: 0.5067
 9984/25000 [==========>...................] - ETA: 32s - loss: 7.5653 - accuracy: 0.5066
10016/25000 [===========>..................] - ETA: 32s - loss: 7.5671 - accuracy: 0.5065
10048/25000 [===========>..................] - ETA: 32s - loss: 7.5674 - accuracy: 0.5065
10080/25000 [===========>..................] - ETA: 32s - loss: 7.5677 - accuracy: 0.5064
10112/25000 [===========>..................] - ETA: 32s - loss: 7.5681 - accuracy: 0.5064
10144/25000 [===========>..................] - ETA: 32s - loss: 7.5638 - accuracy: 0.5067
10176/25000 [===========>..................] - ETA: 32s - loss: 7.5687 - accuracy: 0.5064
10208/25000 [===========>..................] - ETA: 31s - loss: 7.5825 - accuracy: 0.5055
10240/25000 [===========>..................] - ETA: 31s - loss: 7.5903 - accuracy: 0.5050
10272/25000 [===========>..................] - ETA: 31s - loss: 7.5920 - accuracy: 0.5049
10304/25000 [===========>..................] - ETA: 31s - loss: 7.5863 - accuracy: 0.5052
10336/25000 [===========>..................] - ETA: 31s - loss: 7.5850 - accuracy: 0.5053
10368/25000 [===========>..................] - ETA: 31s - loss: 7.5882 - accuracy: 0.5051
10400/25000 [===========>..................] - ETA: 31s - loss: 7.5870 - accuracy: 0.5052
10432/25000 [===========>..................] - ETA: 31s - loss: 7.5843 - accuracy: 0.5054
10464/25000 [===========>..................] - ETA: 31s - loss: 7.5890 - accuracy: 0.5051
10496/25000 [===========>..................] - ETA: 31s - loss: 7.5877 - accuracy: 0.5051
10528/25000 [===========>..................] - ETA: 31s - loss: 7.5880 - accuracy: 0.5051
10560/25000 [===========>..................] - ETA: 31s - loss: 7.5882 - accuracy: 0.5051
10592/25000 [===========>..................] - ETA: 31s - loss: 7.5884 - accuracy: 0.5051
10624/25000 [===========>..................] - ETA: 31s - loss: 7.5858 - accuracy: 0.5053
10656/25000 [===========>..................] - ETA: 30s - loss: 7.5875 - accuracy: 0.5052
10688/25000 [===========>..................] - ETA: 30s - loss: 7.5891 - accuracy: 0.5051
10720/25000 [===========>..................] - ETA: 30s - loss: 7.5865 - accuracy: 0.5052
10752/25000 [===========>..................] - ETA: 30s - loss: 7.5882 - accuracy: 0.5051
10784/25000 [===========>..................] - ETA: 30s - loss: 7.5898 - accuracy: 0.5050
10816/25000 [===========>..................] - ETA: 30s - loss: 7.5915 - accuracy: 0.5049
10848/25000 [============>.................] - ETA: 30s - loss: 7.5889 - accuracy: 0.5051
10880/25000 [============>.................] - ETA: 30s - loss: 7.5905 - accuracy: 0.5050
10912/25000 [============>.................] - ETA: 30s - loss: 7.5936 - accuracy: 0.5048
10944/25000 [============>.................] - ETA: 30s - loss: 7.5910 - accuracy: 0.5049
10976/25000 [============>.................] - ETA: 30s - loss: 7.5926 - accuracy: 0.5048
11008/25000 [============>.................] - ETA: 30s - loss: 7.5984 - accuracy: 0.5045
11040/25000 [============>.................] - ETA: 30s - loss: 7.5972 - accuracy: 0.5045
11072/25000 [============>.................] - ETA: 30s - loss: 7.5974 - accuracy: 0.5045
11104/25000 [============>.................] - ETA: 29s - loss: 7.5962 - accuracy: 0.5046
11136/25000 [============>.................] - ETA: 29s - loss: 7.6019 - accuracy: 0.5042
11168/25000 [============>.................] - ETA: 29s - loss: 7.6007 - accuracy: 0.5043
11200/25000 [============>.................] - ETA: 29s - loss: 7.6036 - accuracy: 0.5041
11232/25000 [============>.................] - ETA: 29s - loss: 7.6106 - accuracy: 0.5037
11264/25000 [============>.................] - ETA: 29s - loss: 7.6163 - accuracy: 0.5033
11296/25000 [============>.................] - ETA: 29s - loss: 7.6178 - accuracy: 0.5032
11328/25000 [============>.................] - ETA: 29s - loss: 7.6220 - accuracy: 0.5029
11360/25000 [============>.................] - ETA: 29s - loss: 7.6248 - accuracy: 0.5027
11392/25000 [============>.................] - ETA: 29s - loss: 7.6235 - accuracy: 0.5028
11424/25000 [============>.................] - ETA: 29s - loss: 7.6196 - accuracy: 0.5031
11456/25000 [============>.................] - ETA: 29s - loss: 7.6198 - accuracy: 0.5031
11488/25000 [============>.................] - ETA: 29s - loss: 7.6172 - accuracy: 0.5032
11520/25000 [============>.................] - ETA: 29s - loss: 7.6147 - accuracy: 0.5034
11552/25000 [============>.................] - ETA: 28s - loss: 7.6122 - accuracy: 0.5035
11584/25000 [============>.................] - ETA: 28s - loss: 7.6190 - accuracy: 0.5031
11616/25000 [============>.................] - ETA: 28s - loss: 7.6204 - accuracy: 0.5030
11648/25000 [============>.................] - ETA: 28s - loss: 7.6192 - accuracy: 0.5031
11680/25000 [=============>................] - ETA: 28s - loss: 7.6180 - accuracy: 0.5032
11712/25000 [=============>................] - ETA: 28s - loss: 7.6247 - accuracy: 0.5027
11744/25000 [=============>................] - ETA: 28s - loss: 7.6301 - accuracy: 0.5024
11776/25000 [=============>................] - ETA: 28s - loss: 7.6289 - accuracy: 0.5025
11808/25000 [=============>................] - ETA: 28s - loss: 7.6264 - accuracy: 0.5026
11840/25000 [=============>................] - ETA: 28s - loss: 7.6226 - accuracy: 0.5029
11872/25000 [=============>................] - ETA: 28s - loss: 7.6292 - accuracy: 0.5024
11904/25000 [=============>................] - ETA: 28s - loss: 7.6331 - accuracy: 0.5022
11936/25000 [=============>................] - ETA: 28s - loss: 7.6371 - accuracy: 0.5019
11968/25000 [=============>................] - ETA: 28s - loss: 7.6372 - accuracy: 0.5019
12000/25000 [=============>................] - ETA: 27s - loss: 7.6385 - accuracy: 0.5018
12032/25000 [=============>................] - ETA: 27s - loss: 7.6335 - accuracy: 0.5022
12064/25000 [=============>................] - ETA: 27s - loss: 7.6298 - accuracy: 0.5024
12096/25000 [=============>................] - ETA: 27s - loss: 7.6299 - accuracy: 0.5024
12128/25000 [=============>................] - ETA: 27s - loss: 7.6287 - accuracy: 0.5025
12160/25000 [=============>................] - ETA: 27s - loss: 7.6237 - accuracy: 0.5028
12192/25000 [=============>................] - ETA: 27s - loss: 7.6213 - accuracy: 0.5030
12224/25000 [=============>................] - ETA: 27s - loss: 7.6252 - accuracy: 0.5027
12256/25000 [=============>................] - ETA: 27s - loss: 7.6253 - accuracy: 0.5027
12288/25000 [=============>................] - ETA: 27s - loss: 7.6304 - accuracy: 0.5024
12320/25000 [=============>................] - ETA: 27s - loss: 7.6280 - accuracy: 0.5025
12352/25000 [=============>................] - ETA: 27s - loss: 7.6294 - accuracy: 0.5024
12384/25000 [=============>................] - ETA: 27s - loss: 7.6295 - accuracy: 0.5024
12416/25000 [=============>................] - ETA: 27s - loss: 7.6234 - accuracy: 0.5028
12448/25000 [=============>................] - ETA: 26s - loss: 7.6297 - accuracy: 0.5024
12480/25000 [=============>................] - ETA: 26s - loss: 7.6298 - accuracy: 0.5024
12512/25000 [==============>...............] - ETA: 26s - loss: 7.6225 - accuracy: 0.5029
12544/25000 [==============>...............] - ETA: 26s - loss: 7.6226 - accuracy: 0.5029
12576/25000 [==============>...............] - ETA: 26s - loss: 7.6239 - accuracy: 0.5028
12608/25000 [==============>...............] - ETA: 26s - loss: 7.6265 - accuracy: 0.5026
12640/25000 [==============>...............] - ETA: 26s - loss: 7.6242 - accuracy: 0.5028
12672/25000 [==============>...............] - ETA: 26s - loss: 7.6231 - accuracy: 0.5028
12704/25000 [==============>...............] - ETA: 26s - loss: 7.6208 - accuracy: 0.5030
12736/25000 [==============>...............] - ETA: 26s - loss: 7.6221 - accuracy: 0.5029
12768/25000 [==============>...............] - ETA: 26s - loss: 7.6234 - accuracy: 0.5028
12800/25000 [==============>...............] - ETA: 26s - loss: 7.6223 - accuracy: 0.5029
12832/25000 [==============>...............] - ETA: 26s - loss: 7.6188 - accuracy: 0.5031
12864/25000 [==============>...............] - ETA: 26s - loss: 7.6177 - accuracy: 0.5032
12896/25000 [==============>...............] - ETA: 25s - loss: 7.6202 - accuracy: 0.5030
12928/25000 [==============>...............] - ETA: 25s - loss: 7.6156 - accuracy: 0.5033
12960/25000 [==============>...............] - ETA: 25s - loss: 7.6146 - accuracy: 0.5034
12992/25000 [==============>...............] - ETA: 25s - loss: 7.6111 - accuracy: 0.5036
13024/25000 [==============>...............] - ETA: 25s - loss: 7.6101 - accuracy: 0.5037
13056/25000 [==============>...............] - ETA: 25s - loss: 7.6138 - accuracy: 0.5034
13088/25000 [==============>...............] - ETA: 25s - loss: 7.6127 - accuracy: 0.5035
13120/25000 [==============>...............] - ETA: 25s - loss: 7.6117 - accuracy: 0.5036
13152/25000 [==============>...............] - ETA: 25s - loss: 7.6142 - accuracy: 0.5034
13184/25000 [==============>...............] - ETA: 25s - loss: 7.6120 - accuracy: 0.5036
13216/25000 [==============>...............] - ETA: 25s - loss: 7.6063 - accuracy: 0.5039
13248/25000 [==============>...............] - ETA: 25s - loss: 7.6030 - accuracy: 0.5042
13280/25000 [==============>...............] - ETA: 25s - loss: 7.5985 - accuracy: 0.5044
13312/25000 [==============>...............] - ETA: 25s - loss: 7.5918 - accuracy: 0.5049
13344/25000 [===============>..............] - ETA: 24s - loss: 7.5954 - accuracy: 0.5046
13376/25000 [===============>..............] - ETA: 24s - loss: 7.6001 - accuracy: 0.5043
13408/25000 [===============>..............] - ETA: 24s - loss: 7.6003 - accuracy: 0.5043
13440/25000 [===============>..............] - ETA: 24s - loss: 7.5970 - accuracy: 0.5045
13472/25000 [===============>..............] - ETA: 24s - loss: 7.5983 - accuracy: 0.5045
13504/25000 [===============>..............] - ETA: 24s - loss: 7.5951 - accuracy: 0.5047
13536/25000 [===============>..............] - ETA: 24s - loss: 7.5953 - accuracy: 0.5047
13568/25000 [===============>..............] - ETA: 24s - loss: 7.5920 - accuracy: 0.5049
13600/25000 [===============>..............] - ETA: 24s - loss: 7.5900 - accuracy: 0.5050
13632/25000 [===============>..............] - ETA: 24s - loss: 7.5890 - accuracy: 0.5051
13664/25000 [===============>..............] - ETA: 24s - loss: 7.5847 - accuracy: 0.5053
13696/25000 [===============>..............] - ETA: 24s - loss: 7.5849 - accuracy: 0.5053
13728/25000 [===============>..............] - ETA: 24s - loss: 7.5828 - accuracy: 0.5055
13760/25000 [===============>..............] - ETA: 24s - loss: 7.5797 - accuracy: 0.5057
13792/25000 [===============>..............] - ETA: 23s - loss: 7.5777 - accuracy: 0.5058
13824/25000 [===============>..............] - ETA: 23s - loss: 7.5790 - accuracy: 0.5057
13856/25000 [===============>..............] - ETA: 23s - loss: 7.5748 - accuracy: 0.5060
13888/25000 [===============>..............] - ETA: 23s - loss: 7.5739 - accuracy: 0.5060
13920/25000 [===============>..............] - ETA: 23s - loss: 7.5686 - accuracy: 0.5064
13952/25000 [===============>..............] - ETA: 23s - loss: 7.5655 - accuracy: 0.5066
13984/25000 [===============>..............] - ETA: 23s - loss: 7.5679 - accuracy: 0.5064
14016/25000 [===============>..............] - ETA: 23s - loss: 7.5693 - accuracy: 0.5063
14048/25000 [===============>..............] - ETA: 23s - loss: 7.5738 - accuracy: 0.5061
14080/25000 [===============>..............] - ETA: 23s - loss: 7.5762 - accuracy: 0.5059
14112/25000 [===============>..............] - ETA: 23s - loss: 7.5797 - accuracy: 0.5057
14144/25000 [===============>..............] - ETA: 23s - loss: 7.5788 - accuracy: 0.5057
14176/25000 [================>.............] - ETA: 23s - loss: 7.5812 - accuracy: 0.5056
14208/25000 [================>.............] - ETA: 23s - loss: 7.5824 - accuracy: 0.5055
14240/25000 [================>.............] - ETA: 22s - loss: 7.5826 - accuracy: 0.5055
14272/25000 [================>.............] - ETA: 22s - loss: 7.5839 - accuracy: 0.5054
14304/25000 [================>.............] - ETA: 22s - loss: 7.5819 - accuracy: 0.5055
14336/25000 [================>.............] - ETA: 22s - loss: 7.5821 - accuracy: 0.5055
14368/25000 [================>.............] - ETA: 22s - loss: 7.5834 - accuracy: 0.5054
14400/25000 [================>.............] - ETA: 22s - loss: 7.5804 - accuracy: 0.5056
14432/25000 [================>.............] - ETA: 22s - loss: 7.5752 - accuracy: 0.5060
14464/25000 [================>.............] - ETA: 22s - loss: 7.5702 - accuracy: 0.5063
14496/25000 [================>.............] - ETA: 22s - loss: 7.5693 - accuracy: 0.5063
14528/25000 [================>.............] - ETA: 22s - loss: 7.5737 - accuracy: 0.5061
14560/25000 [================>.............] - ETA: 22s - loss: 7.5708 - accuracy: 0.5063
14592/25000 [================>.............] - ETA: 22s - loss: 7.5699 - accuracy: 0.5063
14624/25000 [================>.............] - ETA: 22s - loss: 7.5712 - accuracy: 0.5062
14656/25000 [================>.............] - ETA: 22s - loss: 7.5725 - accuracy: 0.5061
14688/25000 [================>.............] - ETA: 21s - loss: 7.5727 - accuracy: 0.5061
14720/25000 [================>.............] - ETA: 21s - loss: 7.5708 - accuracy: 0.5063
14752/25000 [================>.............] - ETA: 21s - loss: 7.5689 - accuracy: 0.5064
14784/25000 [================>.............] - ETA: 21s - loss: 7.5712 - accuracy: 0.5062
14816/25000 [================>.............] - ETA: 21s - loss: 7.5724 - accuracy: 0.5061
14848/25000 [================>.............] - ETA: 21s - loss: 7.5809 - accuracy: 0.5056
14880/25000 [================>.............] - ETA: 21s - loss: 7.5842 - accuracy: 0.5054
14912/25000 [================>.............] - ETA: 21s - loss: 7.5854 - accuracy: 0.5053
14944/25000 [================>.............] - ETA: 21s - loss: 7.5845 - accuracy: 0.5054
14976/25000 [================>.............] - ETA: 21s - loss: 7.5868 - accuracy: 0.5052
15008/25000 [=================>............] - ETA: 21s - loss: 7.5880 - accuracy: 0.5051
15040/25000 [=================>............] - ETA: 21s - loss: 7.5891 - accuracy: 0.5051
15072/25000 [=================>............] - ETA: 21s - loss: 7.5873 - accuracy: 0.5052
15104/25000 [=================>............] - ETA: 21s - loss: 7.5935 - accuracy: 0.5048
15136/25000 [=================>............] - ETA: 20s - loss: 7.5947 - accuracy: 0.5047
15168/25000 [=================>............] - ETA: 20s - loss: 7.5918 - accuracy: 0.5049
15200/25000 [=================>............] - ETA: 20s - loss: 7.5940 - accuracy: 0.5047
15232/25000 [=================>............] - ETA: 20s - loss: 7.5911 - accuracy: 0.5049
15264/25000 [=================>............] - ETA: 20s - loss: 7.5903 - accuracy: 0.5050
15296/25000 [=================>............] - ETA: 20s - loss: 7.5914 - accuracy: 0.5049
15328/25000 [=================>............] - ETA: 20s - loss: 7.5946 - accuracy: 0.5047
15360/25000 [=================>............] - ETA: 20s - loss: 7.5947 - accuracy: 0.5047
15392/25000 [=================>............] - ETA: 20s - loss: 7.5959 - accuracy: 0.5046
15424/25000 [=================>............] - ETA: 20s - loss: 7.5960 - accuracy: 0.5046
15456/25000 [=================>............] - ETA: 20s - loss: 7.5972 - accuracy: 0.5045
15488/25000 [=================>............] - ETA: 20s - loss: 7.5934 - accuracy: 0.5048
15520/25000 [=================>............] - ETA: 20s - loss: 7.5955 - accuracy: 0.5046
15552/25000 [=================>............] - ETA: 20s - loss: 7.5946 - accuracy: 0.5047
15584/25000 [=================>............] - ETA: 20s - loss: 7.5977 - accuracy: 0.5045
15616/25000 [=================>............] - ETA: 19s - loss: 7.5989 - accuracy: 0.5044
15648/25000 [=================>............] - ETA: 19s - loss: 7.6019 - accuracy: 0.5042
15680/25000 [=================>............] - ETA: 19s - loss: 7.6031 - accuracy: 0.5041
15712/25000 [=================>............] - ETA: 19s - loss: 7.6003 - accuracy: 0.5043
15744/25000 [=================>............] - ETA: 19s - loss: 7.5994 - accuracy: 0.5044
15776/25000 [=================>............] - ETA: 19s - loss: 7.5986 - accuracy: 0.5044
15808/25000 [=================>............] - ETA: 19s - loss: 7.5968 - accuracy: 0.5046
15840/25000 [==================>...........] - ETA: 19s - loss: 7.5940 - accuracy: 0.5047
15872/25000 [==================>...........] - ETA: 19s - loss: 7.5922 - accuracy: 0.5049
15904/25000 [==================>...........] - ETA: 19s - loss: 7.5895 - accuracy: 0.5050
15936/25000 [==================>...........] - ETA: 19s - loss: 7.5916 - accuracy: 0.5049
15968/25000 [==================>...........] - ETA: 19s - loss: 7.5927 - accuracy: 0.5048
16000/25000 [==================>...........] - ETA: 19s - loss: 7.5909 - accuracy: 0.5049
16032/25000 [==================>...........] - ETA: 19s - loss: 7.5872 - accuracy: 0.5052
16064/25000 [==================>...........] - ETA: 18s - loss: 7.5874 - accuracy: 0.5052
16096/25000 [==================>...........] - ETA: 18s - loss: 7.5837 - accuracy: 0.5054
16128/25000 [==================>...........] - ETA: 18s - loss: 7.5858 - accuracy: 0.5053
16160/25000 [==================>...........] - ETA: 18s - loss: 7.5907 - accuracy: 0.5050
16192/25000 [==================>...........] - ETA: 18s - loss: 7.5918 - accuracy: 0.5049
16224/25000 [==================>...........] - ETA: 18s - loss: 7.5929 - accuracy: 0.5048
16256/25000 [==================>...........] - ETA: 18s - loss: 7.5949 - accuracy: 0.5047
16288/25000 [==================>...........] - ETA: 18s - loss: 7.5941 - accuracy: 0.5047
16320/25000 [==================>...........] - ETA: 18s - loss: 7.5962 - accuracy: 0.5046
16352/25000 [==================>...........] - ETA: 18s - loss: 7.5963 - accuracy: 0.5046
16384/25000 [==================>...........] - ETA: 18s - loss: 7.5955 - accuracy: 0.5046
16416/25000 [==================>...........] - ETA: 18s - loss: 7.5900 - accuracy: 0.5050
16448/25000 [==================>...........] - ETA: 18s - loss: 7.5920 - accuracy: 0.5049
16480/25000 [==================>...........] - ETA: 18s - loss: 7.5903 - accuracy: 0.5050
16512/25000 [==================>...........] - ETA: 18s - loss: 7.5923 - accuracy: 0.5048
16544/25000 [==================>...........] - ETA: 17s - loss: 7.5888 - accuracy: 0.5051
16576/25000 [==================>...........] - ETA: 17s - loss: 7.5954 - accuracy: 0.5046
16608/25000 [==================>...........] - ETA: 17s - loss: 7.5965 - accuracy: 0.5046
16640/25000 [==================>...........] - ETA: 17s - loss: 7.5984 - accuracy: 0.5044
16672/25000 [===================>..........] - ETA: 17s - loss: 7.5986 - accuracy: 0.5044
16704/25000 [===================>..........] - ETA: 17s - loss: 7.5996 - accuracy: 0.5044
16736/25000 [===================>..........] - ETA: 17s - loss: 7.6007 - accuracy: 0.5043
16768/25000 [===================>..........] - ETA: 17s - loss: 7.6026 - accuracy: 0.5042
16800/25000 [===================>..........] - ETA: 17s - loss: 7.6018 - accuracy: 0.5042
16832/25000 [===================>..........] - ETA: 17s - loss: 7.6029 - accuracy: 0.5042
16864/25000 [===================>..........] - ETA: 17s - loss: 7.6039 - accuracy: 0.5041
16896/25000 [===================>..........] - ETA: 17s - loss: 7.6067 - accuracy: 0.5039
16928/25000 [===================>..........] - ETA: 17s - loss: 7.6077 - accuracy: 0.5038
16960/25000 [===================>..........] - ETA: 17s - loss: 7.6106 - accuracy: 0.5037
16992/25000 [===================>..........] - ETA: 16s - loss: 7.6134 - accuracy: 0.5035
17024/25000 [===================>..........] - ETA: 16s - loss: 7.6117 - accuracy: 0.5036
17056/25000 [===================>..........] - ETA: 16s - loss: 7.6154 - accuracy: 0.5033
17088/25000 [===================>..........] - ETA: 16s - loss: 7.6128 - accuracy: 0.5035
17120/25000 [===================>..........] - ETA: 16s - loss: 7.6129 - accuracy: 0.5035
17152/25000 [===================>..........] - ETA: 16s - loss: 7.6103 - accuracy: 0.5037
17184/25000 [===================>..........] - ETA: 16s - loss: 7.6104 - accuracy: 0.5037
17216/25000 [===================>..........] - ETA: 16s - loss: 7.6114 - accuracy: 0.5036
17248/25000 [===================>..........] - ETA: 16s - loss: 7.6097 - accuracy: 0.5037
17280/25000 [===================>..........] - ETA: 16s - loss: 7.6107 - accuracy: 0.5036
17312/25000 [===================>..........] - ETA: 16s - loss: 7.6117 - accuracy: 0.5036
17344/25000 [===================>..........] - ETA: 16s - loss: 7.6100 - accuracy: 0.5037
17376/25000 [===================>..........] - ETA: 16s - loss: 7.6119 - accuracy: 0.5036
17408/25000 [===================>..........] - ETA: 16s - loss: 7.6155 - accuracy: 0.5033
17440/25000 [===================>..........] - ETA: 15s - loss: 7.6174 - accuracy: 0.5032
17472/25000 [===================>..........] - ETA: 15s - loss: 7.6201 - accuracy: 0.5030
17504/25000 [====================>.........] - ETA: 15s - loss: 7.6219 - accuracy: 0.5029
17536/25000 [====================>.........] - ETA: 15s - loss: 7.6238 - accuracy: 0.5028
17568/25000 [====================>.........] - ETA: 15s - loss: 7.6256 - accuracy: 0.5027
17600/25000 [====================>.........] - ETA: 15s - loss: 7.6283 - accuracy: 0.5025
17632/25000 [====================>.........] - ETA: 15s - loss: 7.6344 - accuracy: 0.5021
17664/25000 [====================>.........] - ETA: 15s - loss: 7.6354 - accuracy: 0.5020
17696/25000 [====================>.........] - ETA: 15s - loss: 7.6346 - accuracy: 0.5021
17728/25000 [====================>.........] - ETA: 15s - loss: 7.6294 - accuracy: 0.5024
17760/25000 [====================>.........] - ETA: 15s - loss: 7.6269 - accuracy: 0.5026
17792/25000 [====================>.........] - ETA: 15s - loss: 7.6253 - accuracy: 0.5027
17824/25000 [====================>.........] - ETA: 15s - loss: 7.6262 - accuracy: 0.5026
17856/25000 [====================>.........] - ETA: 15s - loss: 7.6271 - accuracy: 0.5026
17888/25000 [====================>.........] - ETA: 15s - loss: 7.6280 - accuracy: 0.5025
17920/25000 [====================>.........] - ETA: 14s - loss: 7.6290 - accuracy: 0.5025
17952/25000 [====================>.........] - ETA: 14s - loss: 7.6299 - accuracy: 0.5024
17984/25000 [====================>.........] - ETA: 14s - loss: 7.6325 - accuracy: 0.5022
18016/25000 [====================>.........] - ETA: 14s - loss: 7.6368 - accuracy: 0.5019
18048/25000 [====================>.........] - ETA: 14s - loss: 7.6420 - accuracy: 0.5016
18080/25000 [====================>.........] - ETA: 14s - loss: 7.6429 - accuracy: 0.5015
18112/25000 [====================>.........] - ETA: 14s - loss: 7.6438 - accuracy: 0.5015
18144/25000 [====================>.........] - ETA: 14s - loss: 7.6472 - accuracy: 0.5013
18176/25000 [====================>.........] - ETA: 14s - loss: 7.6472 - accuracy: 0.5013
18208/25000 [====================>.........] - ETA: 14s - loss: 7.6464 - accuracy: 0.5013
18240/25000 [====================>.........] - ETA: 14s - loss: 7.6481 - accuracy: 0.5012
18272/25000 [====================>.........] - ETA: 14s - loss: 7.6490 - accuracy: 0.5011
18304/25000 [====================>.........] - ETA: 14s - loss: 7.6474 - accuracy: 0.5013
18336/25000 [=====================>........] - ETA: 14s - loss: 7.6449 - accuracy: 0.5014
18368/25000 [=====================>........] - ETA: 13s - loss: 7.6483 - accuracy: 0.5012
18400/25000 [=====================>........] - ETA: 13s - loss: 7.6483 - accuracy: 0.5012
18432/25000 [=====================>........] - ETA: 13s - loss: 7.6491 - accuracy: 0.5011
18464/25000 [=====================>........] - ETA: 13s - loss: 7.6500 - accuracy: 0.5011
18496/25000 [=====================>........] - ETA: 13s - loss: 7.6484 - accuracy: 0.5012
18528/25000 [=====================>........] - ETA: 13s - loss: 7.6492 - accuracy: 0.5011
18560/25000 [=====================>........] - ETA: 13s - loss: 7.6493 - accuracy: 0.5011
18592/25000 [=====================>........] - ETA: 13s - loss: 7.6509 - accuracy: 0.5010
18624/25000 [=====================>........] - ETA: 13s - loss: 7.6493 - accuracy: 0.5011
18656/25000 [=====================>........] - ETA: 13s - loss: 7.6494 - accuracy: 0.5011
18688/25000 [=====================>........] - ETA: 13s - loss: 7.6527 - accuracy: 0.5009
18720/25000 [=====================>........] - ETA: 13s - loss: 7.6527 - accuracy: 0.5009
18752/25000 [=====================>........] - ETA: 13s - loss: 7.6527 - accuracy: 0.5009
18784/25000 [=====================>........] - ETA: 13s - loss: 7.6511 - accuracy: 0.5010
18816/25000 [=====================>........] - ETA: 13s - loss: 7.6544 - accuracy: 0.5008
18848/25000 [=====================>........] - ETA: 12s - loss: 7.6528 - accuracy: 0.5009
18880/25000 [=====================>........] - ETA: 12s - loss: 7.6528 - accuracy: 0.5009
18912/25000 [=====================>........] - ETA: 12s - loss: 7.6512 - accuracy: 0.5010
18944/25000 [=====================>........] - ETA: 12s - loss: 7.6521 - accuracy: 0.5010
18976/25000 [=====================>........] - ETA: 12s - loss: 7.6513 - accuracy: 0.5010
19008/25000 [=====================>........] - ETA: 12s - loss: 7.6521 - accuracy: 0.5009
19040/25000 [=====================>........] - ETA: 12s - loss: 7.6505 - accuracy: 0.5011
19072/25000 [=====================>........] - ETA: 12s - loss: 7.6497 - accuracy: 0.5011
19104/25000 [=====================>........] - ETA: 12s - loss: 7.6498 - accuracy: 0.5011
19136/25000 [=====================>........] - ETA: 12s - loss: 7.6506 - accuracy: 0.5010
19168/25000 [======================>.......] - ETA: 12s - loss: 7.6530 - accuracy: 0.5009
19200/25000 [======================>.......] - ETA: 12s - loss: 7.6538 - accuracy: 0.5008
19232/25000 [======================>.......] - ETA: 12s - loss: 7.6483 - accuracy: 0.5012
19264/25000 [======================>.......] - ETA: 12s - loss: 7.6475 - accuracy: 0.5012
19296/25000 [======================>.......] - ETA: 12s - loss: 7.6460 - accuracy: 0.5013
19328/25000 [======================>.......] - ETA: 11s - loss: 7.6420 - accuracy: 0.5016
19360/25000 [======================>.......] - ETA: 11s - loss: 7.6405 - accuracy: 0.5017
19392/25000 [======================>.......] - ETA: 11s - loss: 7.6421 - accuracy: 0.5016
19424/25000 [======================>.......] - ETA: 11s - loss: 7.6406 - accuracy: 0.5017
19456/25000 [======================>.......] - ETA: 11s - loss: 7.6438 - accuracy: 0.5015
19488/25000 [======================>.......] - ETA: 11s - loss: 7.6438 - accuracy: 0.5015
19520/25000 [======================>.......] - ETA: 11s - loss: 7.6431 - accuracy: 0.5015
19552/25000 [======================>.......] - ETA: 11s - loss: 7.6423 - accuracy: 0.5016
19584/25000 [======================>.......] - ETA: 11s - loss: 7.6423 - accuracy: 0.5016
19616/25000 [======================>.......] - ETA: 11s - loss: 7.6424 - accuracy: 0.5016
19648/25000 [======================>.......] - ETA: 11s - loss: 7.6432 - accuracy: 0.5015
19680/25000 [======================>.......] - ETA: 11s - loss: 7.6409 - accuracy: 0.5017
19712/25000 [======================>.......] - ETA: 11s - loss: 7.6386 - accuracy: 0.5018
19744/25000 [======================>.......] - ETA: 11s - loss: 7.6402 - accuracy: 0.5017
19776/25000 [======================>.......] - ETA: 11s - loss: 7.6410 - accuracy: 0.5017
19808/25000 [======================>.......] - ETA: 10s - loss: 7.6372 - accuracy: 0.5019
19840/25000 [======================>.......] - ETA: 10s - loss: 7.6342 - accuracy: 0.5021
19872/25000 [======================>.......] - ETA: 10s - loss: 7.6350 - accuracy: 0.5021
19904/25000 [======================>.......] - ETA: 10s - loss: 7.6366 - accuracy: 0.5020
19936/25000 [======================>.......] - ETA: 10s - loss: 7.6366 - accuracy: 0.5020
19968/25000 [======================>.......] - ETA: 10s - loss: 7.6397 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 10s - loss: 7.6360 - accuracy: 0.5020
20032/25000 [=======================>......] - ETA: 10s - loss: 7.6368 - accuracy: 0.5019
20064/25000 [=======================>......] - ETA: 10s - loss: 7.6330 - accuracy: 0.5022
20096/25000 [=======================>......] - ETA: 10s - loss: 7.6361 - accuracy: 0.5020
20128/25000 [=======================>......] - ETA: 10s - loss: 7.6384 - accuracy: 0.5018
20160/25000 [=======================>......] - ETA: 10s - loss: 7.6347 - accuracy: 0.5021
20192/25000 [=======================>......] - ETA: 10s - loss: 7.6340 - accuracy: 0.5021
20224/25000 [=======================>......] - ETA: 10s - loss: 7.6340 - accuracy: 0.5021
20256/25000 [=======================>......] - ETA: 10s - loss: 7.6333 - accuracy: 0.5022
20288/25000 [=======================>......] - ETA: 9s - loss: 7.6326 - accuracy: 0.5022 
20320/25000 [=======================>......] - ETA: 9s - loss: 7.6364 - accuracy: 0.5020
20352/25000 [=======================>......] - ETA: 9s - loss: 7.6342 - accuracy: 0.5021
20384/25000 [=======================>......] - ETA: 9s - loss: 7.6350 - accuracy: 0.5021
20416/25000 [=======================>......] - ETA: 9s - loss: 7.6388 - accuracy: 0.5018
20448/25000 [=======================>......] - ETA: 9s - loss: 7.6359 - accuracy: 0.5020
20480/25000 [=======================>......] - ETA: 9s - loss: 7.6382 - accuracy: 0.5019
20512/25000 [=======================>......] - ETA: 9s - loss: 7.6367 - accuracy: 0.5020
20544/25000 [=======================>......] - ETA: 9s - loss: 7.6390 - accuracy: 0.5018
20576/25000 [=======================>......] - ETA: 9s - loss: 7.6368 - accuracy: 0.5019
20608/25000 [=======================>......] - ETA: 9s - loss: 7.6376 - accuracy: 0.5019
20640/25000 [=======================>......] - ETA: 9s - loss: 7.6384 - accuracy: 0.5018
20672/25000 [=======================>......] - ETA: 9s - loss: 7.6362 - accuracy: 0.5020
20704/25000 [=======================>......] - ETA: 9s - loss: 7.6348 - accuracy: 0.5021
20736/25000 [=======================>......] - ETA: 8s - loss: 7.6348 - accuracy: 0.5021
20768/25000 [=======================>......] - ETA: 8s - loss: 7.6371 - accuracy: 0.5019
20800/25000 [=======================>......] - ETA: 8s - loss: 7.6349 - accuracy: 0.5021
20832/25000 [=======================>......] - ETA: 8s - loss: 7.6306 - accuracy: 0.5024
20864/25000 [========================>.....] - ETA: 8s - loss: 7.6343 - accuracy: 0.5021
20896/25000 [========================>.....] - ETA: 8s - loss: 7.6351 - accuracy: 0.5021
20928/25000 [========================>.....] - ETA: 8s - loss: 7.6344 - accuracy: 0.5021
20960/25000 [========================>.....] - ETA: 8s - loss: 7.6300 - accuracy: 0.5024
20992/25000 [========================>.....] - ETA: 8s - loss: 7.6250 - accuracy: 0.5027
21024/25000 [========================>.....] - ETA: 8s - loss: 7.6221 - accuracy: 0.5029
21056/25000 [========================>.....] - ETA: 8s - loss: 7.6222 - accuracy: 0.5029
21088/25000 [========================>.....] - ETA: 8s - loss: 7.6230 - accuracy: 0.5028
21120/25000 [========================>.....] - ETA: 8s - loss: 7.6260 - accuracy: 0.5027
21152/25000 [========================>.....] - ETA: 8s - loss: 7.6260 - accuracy: 0.5026
21184/25000 [========================>.....] - ETA: 8s - loss: 7.6283 - accuracy: 0.5025
21216/25000 [========================>.....] - ETA: 7s - loss: 7.6283 - accuracy: 0.5025
21248/25000 [========================>.....] - ETA: 7s - loss: 7.6298 - accuracy: 0.5024
21280/25000 [========================>.....] - ETA: 7s - loss: 7.6313 - accuracy: 0.5023
21312/25000 [========================>.....] - ETA: 7s - loss: 7.6328 - accuracy: 0.5022
21344/25000 [========================>.....] - ETA: 7s - loss: 7.6336 - accuracy: 0.5022
21376/25000 [========================>.....] - ETA: 7s - loss: 7.6336 - accuracy: 0.5022
21408/25000 [========================>.....] - ETA: 7s - loss: 7.6351 - accuracy: 0.5021
21440/25000 [========================>.....] - ETA: 7s - loss: 7.6337 - accuracy: 0.5021
21472/25000 [========================>.....] - ETA: 7s - loss: 7.6359 - accuracy: 0.5020
21504/25000 [========================>.....] - ETA: 7s - loss: 7.6338 - accuracy: 0.5021
21536/25000 [========================>.....] - ETA: 7s - loss: 7.6346 - accuracy: 0.5021
21568/25000 [========================>.....] - ETA: 7s - loss: 7.6375 - accuracy: 0.5019
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6382 - accuracy: 0.5019
21632/25000 [========================>.....] - ETA: 7s - loss: 7.6404 - accuracy: 0.5017
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6397 - accuracy: 0.5018
21696/25000 [=========================>....] - ETA: 6s - loss: 7.6412 - accuracy: 0.5017
21728/25000 [=========================>....] - ETA: 6s - loss: 7.6440 - accuracy: 0.5015
21760/25000 [=========================>....] - ETA: 6s - loss: 7.6441 - accuracy: 0.5015
21792/25000 [=========================>....] - ETA: 6s - loss: 7.6455 - accuracy: 0.5014
21824/25000 [=========================>....] - ETA: 6s - loss: 7.6441 - accuracy: 0.5015
21856/25000 [=========================>....] - ETA: 6s - loss: 7.6456 - accuracy: 0.5014
21888/25000 [=========================>....] - ETA: 6s - loss: 7.6442 - accuracy: 0.5015
21920/25000 [=========================>....] - ETA: 6s - loss: 7.6428 - accuracy: 0.5016
21952/25000 [=========================>....] - ETA: 6s - loss: 7.6401 - accuracy: 0.5017
21984/25000 [=========================>....] - ETA: 6s - loss: 7.6380 - accuracy: 0.5019
22016/25000 [=========================>....] - ETA: 6s - loss: 7.6388 - accuracy: 0.5018
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6339 - accuracy: 0.5021
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6381 - accuracy: 0.5019
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6375 - accuracy: 0.5019
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6341 - accuracy: 0.5021
22176/25000 [=========================>....] - ETA: 5s - loss: 7.6327 - accuracy: 0.5022
22208/25000 [=========================>....] - ETA: 5s - loss: 7.6349 - accuracy: 0.5021
22240/25000 [=========================>....] - ETA: 5s - loss: 7.6349 - accuracy: 0.5021
22272/25000 [=========================>....] - ETA: 5s - loss: 7.6391 - accuracy: 0.5018
22304/25000 [=========================>....] - ETA: 5s - loss: 7.6419 - accuracy: 0.5016
22336/25000 [=========================>....] - ETA: 5s - loss: 7.6412 - accuracy: 0.5017
22368/25000 [=========================>....] - ETA: 5s - loss: 7.6385 - accuracy: 0.5018
22400/25000 [=========================>....] - ETA: 5s - loss: 7.6386 - accuracy: 0.5018
22432/25000 [=========================>....] - ETA: 5s - loss: 7.6413 - accuracy: 0.5016
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6420 - accuracy: 0.5016
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6407 - accuracy: 0.5017
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6421 - accuracy: 0.5016
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6449 - accuracy: 0.5014
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6456 - accuracy: 0.5014
22656/25000 [==========================>...] - ETA: 4s - loss: 7.6450 - accuracy: 0.5014
22688/25000 [==========================>...] - ETA: 4s - loss: 7.6477 - accuracy: 0.5012
22720/25000 [==========================>...] - ETA: 4s - loss: 7.6484 - accuracy: 0.5012
22752/25000 [==========================>...] - ETA: 4s - loss: 7.6484 - accuracy: 0.5012
22784/25000 [==========================>...] - ETA: 4s - loss: 7.6511 - accuracy: 0.5010
22816/25000 [==========================>...] - ETA: 4s - loss: 7.6525 - accuracy: 0.5009
22848/25000 [==========================>...] - ETA: 4s - loss: 7.6512 - accuracy: 0.5010
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6525 - accuracy: 0.5009
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6552 - accuracy: 0.5007
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6533 - accuracy: 0.5009
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6540 - accuracy: 0.5008
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6547 - accuracy: 0.5008
23104/25000 [==========================>...] - ETA: 3s - loss: 7.6606 - accuracy: 0.5004
23136/25000 [==========================>...] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
23168/25000 [==========================>...] - ETA: 3s - loss: 7.6613 - accuracy: 0.5003
23200/25000 [==========================>...] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
23232/25000 [==========================>...] - ETA: 3s - loss: 7.6594 - accuracy: 0.5005
23264/25000 [==========================>...] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6607 - accuracy: 0.5004
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6581 - accuracy: 0.5006
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6607 - accuracy: 0.5004
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6601 - accuracy: 0.5004
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6601 - accuracy: 0.5004
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6594 - accuracy: 0.5005
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6594 - accuracy: 0.5005
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23584/25000 [===========================>..] - ETA: 2s - loss: 7.6536 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 2s - loss: 7.6523 - accuracy: 0.5009
23648/25000 [===========================>..] - ETA: 2s - loss: 7.6517 - accuracy: 0.5010
23680/25000 [===========================>..] - ETA: 2s - loss: 7.6530 - accuracy: 0.5009
23712/25000 [===========================>..] - ETA: 2s - loss: 7.6511 - accuracy: 0.5010
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6524 - accuracy: 0.5009
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6531 - accuracy: 0.5009
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6525 - accuracy: 0.5009
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6512 - accuracy: 0.5010
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6506 - accuracy: 0.5010
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6493 - accuracy: 0.5011
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6512 - accuracy: 0.5010
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6487 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6494 - accuracy: 0.5011
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
24064/25000 [===========================>..] - ETA: 1s - loss: 7.6526 - accuracy: 0.5009
24096/25000 [===========================>..] - ETA: 1s - loss: 7.6533 - accuracy: 0.5009
24128/25000 [===========================>..] - ETA: 1s - loss: 7.6533 - accuracy: 0.5009
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6501 - accuracy: 0.5011
24192/25000 [============================>.] - ETA: 1s - loss: 7.6520 - accuracy: 0.5010
24224/25000 [============================>.] - ETA: 1s - loss: 7.6514 - accuracy: 0.5010
24256/25000 [============================>.] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
24288/25000 [============================>.] - ETA: 1s - loss: 7.6571 - accuracy: 0.5006
24320/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24352/25000 [============================>.] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24384/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24416/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24448/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
24480/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24512/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24544/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24576/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24608/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24640/25000 [============================>.] - ETA: 0s - loss: 7.6616 - accuracy: 0.5003
24672/25000 [============================>.] - ETA: 0s - loss: 7.6616 - accuracy: 0.5003
24704/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24736/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24832/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 62s 2ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
Loading data...
Using TensorFlow backend.





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;34m[0m[0m
[0;32m---> 10[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'os' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: tuple index out of range

During handling of the above exception, another exception occurred:

[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mmodel[0m         [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mModel[0m[0;34m([0m[0mmodel_pars[0m[0;34m=[0m[0mmodel_pars[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m)[0m             [0;31m# Create Model instance[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mmodel[0m[0;34m,[0m [0msess[0m   [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m,[0m [0mout_pars[0m[0;34m=[0m[0mout_pars[0m[0;34m)[0m          [0;31m# fit the model[0m[0;34m[0m[0;34m[0m[0m

[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_home_retail.ipynb 

Deprecaton set to False
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0mdata_path[0m [0;34m=[0m [0;34m'hyper_lightgbm_home_retail.json'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vison_fashion_MNIST.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//mnist_mlmodels_.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras_charcnn_reuters.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mmodel_pars[0m      [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'model_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mdata_pars[0m       [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'data_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mcompute_pars[0m    [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'compute_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mout_pars[0m        [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'out_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'reuters_charcnn.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow__lstm_json.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'os' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow_1_lstm.ipynb 

/home/runner/work/mlmodels/mlmodels
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
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
test

  #### Module init   ############################################ 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  #### Model init   ############################################ 

  <mlmodels.model_tf.1_lstm.Model object at 0x7f127ca819e8> 

  #### Fit   ######################################################## 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
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

  #### Predict   #################################################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
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
[[ 0.          0.          0.          0.          0.          0.        ]
 [ 0.15133244  0.05933547 -0.05320736  0.04452814  0.04251817  0.0734596 ]
 [-0.06147702  0.07839675 -0.04043624 -0.20941538 -0.13038447  0.21403019]
 [ 0.04386913  0.1800684   0.44488692  0.05423185  0.11182792  0.17470102]
 [-0.22959796  0.10426589  0.25868329  0.03994475  0.03219848  0.30151519]
 [-0.07632279  0.24884845  0.1741472  -0.20755182  0.03282016  0.18690975]
 [-0.56780517  0.28767392  0.90407634  0.22998625 -0.18638329 -0.47090808]
 [ 0.20653346  0.16490579  0.18010314 -0.07662384 -0.44649598  0.56727856]
 [ 0.62988609  0.27441457 -0.35819545 -0.17501149  0.22796017  0.57498878]
 [ 0.          0.          0.          0.          0.          0.        ]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 
model_tf/1_lstm.py
model_tf.1_lstm.py
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'} 

  #### Loading dataset   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]

  #### Model init  ############################################# 

  #### Model fit   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
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

  #### Predict   ##################################################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
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

  #### metrics   ##################################################### 
{'loss': 0.4921247884631157, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-22 08:53:52.373372: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 458, in test_cli
    test(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 189, in test
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()

model_tf/1_lstm.py
model_tf.1_lstm.py
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'} 

  #### Loading dataset   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]

  #### Model init  ############################################# 

  #### Model fit   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
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

  #### Predict   ##################################################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
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

  #### metrics   ##################################################### 
{'loss': 0.4618277996778488, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-22 08:53:53.317977: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 460, in test_cli
    test_global(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 200, in test_global
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//timeseries_m5_deepar.ipynb 

UsageError: Line magic function `%%capture` not found.





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_svm.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: tuple index out of range

During handling of the above exception, another exception occurred:

[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={

[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_titanic.ipynb 

Deprecaton set to False
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0mdata_path[0m [0;34m=[0m [0;34m'hyper_lightgbm_titanic.json'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//fashion_MNIST_mlmodels.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb 

Deprecaton set to False
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'../mlmodels/dataset/json/hyper_titanic_randomForest.json'[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: tuple index out of range

During handling of the above exception, another exception occurred:

[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={

[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl_titanic.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m     [0mchoice[0m[0;34m=[0m[0;34m'json'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0mconfig_mode[0m[0;34m=[0m [0;34m'test'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m     [0mdata_path[0m[0;34m=[0m [0;34m'../mlmodels/dataset/json/gluon_automl.json'[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m )

[0;32m~/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py[0m in [0;36mget_params[0;34m(choice, data_path, config_mode, **kw)[0m
[1;32m     80[0m             __file__)).parent.parent / "model_gluon/gluon_automl.json" if data_path == "dataset/" else data_path
[1;32m     81[0m [0;34m[0m[0m
[0;32m---> 82[0;31m         [0;32mwith[0m [0mopen[0m[0;34m([0m[0mdata_path[0m[0;34m,[0m [0mencoding[0m[0;34m=[0m[0;34m'utf-8'[0m[0;34m)[0m [0;32mas[0m [0mconfig_f[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m             [0mconfig[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mconfig_f[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     84[0m             [0mconfig[0m [0;34m=[0m [0mconfig[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'lightgbm_titanic.json'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m
[0;32m----> 6[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'lightgbm_titanic.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_model.py 

<module 'mlmodels' from '/home/runner/work/mlmodels/mlmodels/mlmodels/__init__.py'>
/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/ardmn.json
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_model.py[0m in [0;36m<module>[0;34m[0m
[1;32m     25[0m [0;31m# Model Parameters[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m [0;31m# model_pars, data_pars, compute_pars, out_pars[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpdict[0m   [0;34m)[0m   [0;31m###Normalize path[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/ardmn.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
[0;31m    !git clone https://github.com/ahmed3bbas/mlmodels.git[0m
[0m    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m5.py 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py[0m in [0;36m<module>[0;34m[0m
[1;32m     84[0m [0;34m[0m[0m
[1;32m     85[0m """
[0;32m---> 86[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     87[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     88[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36mparser_f[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)[0m
[1;32m    683[0m         )
[1;32m    684[0m [0;34m[0m[0m
[0;32m--> 685[0;31m         [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    686[0m [0;34m[0m[0m
[1;32m    687[0m     [0mparser_f[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    455[0m [0;34m[0m[0m
[1;32m    456[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 457[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfp_or_buf[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    458[0m [0;34m[0m[0m
[1;32m    459[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m    893[0m             [0mself[0m[0;34m.[0m[0moptions[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m [0;34m=[0m [0mkwds[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    894[0m [0;34m[0m[0m
[0;32m--> 895[0;31m         [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    896[0m [0;34m[0m[0m
[1;32m    897[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_make_engine[0;34m(self, engine)[0m
[1;32m   1133[0m     [0;32mdef[0m [0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"c"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1134[0m         [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"c"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1135[0;31m             [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mCParserWrapper[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1136[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1137[0m             [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
[1;32m   1915[0m         [0mkwds[0m[0;34m[[0m[0;34m"usecols"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0musecols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1916[0m [0;34m[0m[0m
[0;32m-> 1917[0;31m         [0mself[0m[0;34m.[0m[0m_reader[0m [0;34m=[0m [0mparsers[0m[0;34m.[0m[0mTextReader[0m[0;34m([0m[0msrc[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1918[0m         [0mself[0m[0;34m.[0m[0munnamed_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reader[0m[0;34m.[0m[0munnamed_cols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1919[0m [0;34m[0m[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader.__cinit__[0;34m()[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader._setup_parser_source[0;34m()[0m

[0;31mFileNotFoundError[0m: [Errno 2] File b'./m5-forecasting-accuracy/calendar.csv' does not exist: b'./m5-forecasting-accuracy/calendar.csv'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_hyper.py 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_hyper.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mutil[0m [0;32mimport[0m [0mpath_norm_dict[0m[0;34m,[0m [0mpath_norm[0m[0;34m,[0m [0mparams_json_load[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mprint[0m[0;34m([0m[0mmlmodels[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m

[0;31mNameError[0m: name 'mlmodels' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m4.py 






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.py 

Deprecaton set to False
/home/runner/work/mlmodels/mlmodels
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/lightgbm_glass.py[0m in [0;36m<module>[0;34m[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m
[0;32m---> 22[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m [0mprint[0m[0;34m([0m[0mpars[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'lightgbm_glass.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m5.py 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py[0m in [0;36m<module>[0;34m[0m
[1;32m     84[0m [0;34m[0m[0m
[1;32m     85[0m """
[0;32m---> 86[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     87[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     88[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36mparser_f[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)[0m
[1;32m    683[0m         )
[1;32m    684[0m [0;34m[0m[0m
[0;32m--> 685[0;31m         [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    686[0m [0;34m[0m[0m
[1;32m    687[0m     [0mparser_f[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    455[0m [0;34m[0m[0m
[1;32m    456[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 457[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfp_or_buf[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    458[0m [0;34m[0m[0m
[1;32m    459[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m    893[0m             [0mself[0m[0;34m.[0m[0moptions[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m [0;34m=[0m [0mkwds[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    894[0m [0;34m[0m[0m
[0;32m--> 895[0;31m         [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    896[0m [0;34m[0m[0m
[1;32m    897[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_make_engine[0;34m(self, engine)[0m
[1;32m   1133[0m     [0;32mdef[0m [0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"c"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1134[0m         [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"c"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1135[0;31m             [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mCParserWrapper[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1136[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1137[0m             [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
[1;32m   1915[0m         [0mkwds[0m[0;34m[[0m[0;34m"usecols"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0musecols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1916[0m [0;34m[0m[0m
[0;32m-> 1917[0;31m         [0mself[0m[0;34m.[0m[0m_reader[0m [0;34m=[0m [0mparsers[0m[0;34m.[0m[0mTextReader[0m[0;34m([0m[0msrc[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1918[0m         [0mself[0m[0;34m.[0m[0munnamed_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reader[0m[0;34m.[0m[0munnamed_cols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1919[0m [0;34m[0m[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader.__cinit__[0;34m()[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader._setup_parser_source[0;34m()[0m

[0;31mFileNotFoundError[0m: [Errno 2] File b'./m5-forecasting-accuracy/calendar.csv' does not exist: b'./m5-forecasting-accuracy/calendar.csv'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m4.py 


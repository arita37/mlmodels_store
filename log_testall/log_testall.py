
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '82ca0cabe4779c98bad687c53f6357fc6efdf783', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/82ca0cabe4779c98bad687c53f6357fc6efdf783

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_dev.temporal_fusion_google', 'model_tf.1_lstm', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model_old', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.torchhub', 'model_tch.transformer_sentence', 'model_tch.textcnn'] 

  Used ['model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_dev.temporal_fusion_google', 'model_tf.1_lstm', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model_old', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.torchhub', 'model_tch.transformer_sentence', 'model_tch.textcnn'] 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 261, in <module>
    test(pars_choice="json", data_path=f"dataset/json/refactor/charcnn_zhang.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 222, in test
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 151, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/charcnn_zhang.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 270e28b] ml_store
 2 files changed, 79 insertions(+), 10827 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + e200f0e...270e28b master -> master (forced update)





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 373, in <module>
    test(pars_choice="json", data_path= f"dataset/json/refactor/charcnn.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 330, in test
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 266, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/charcnn.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master aa918a4] ml_store
 1 file changed, 43 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   270e28b..aa918a4  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.namentity_crm_bilstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py'> 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 306, in <module>
    test_module(model_uri=MODEL_URI, param_pars=param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 197, in get_params
    cf = json.load(open(data_path, mode="r"))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 80fbeb8] ml_store
 1 file changed, 46 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   aa918a4..80fbeb8  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py", line 258, in <module>
    test_module(model_uri = MODEL_URI, param_pars= param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py", line 165, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn_keras.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 5f88d29] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   80fbeb8..5f88d29  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_dev//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_dev//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 4f4adee] ml_store
 1 file changed, 34 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5f88d29..4f4adee  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
start

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
{'loss': 0.4228816367685795, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 31b6817] ml_store
 1 file changed, 112 insertions(+)
To github.com:arita37/mlmodels_store.git
   4f4adee..31b6817  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluon_automl.py 

  #### Loading params   ############################################## 

  #### Model params   ################################################ 

  #### Loading dataset   ############################################# 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073

  #### Model init, fit   ############################################# 
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
num_leaves:   Int: lower=26, upper=66
learning_rate:   Real: lower=0.005, upper=0.2
feature_fraction:   Real: lower=0.75, upper=1.0
min_data_in_leaf:   Int: lower=2, upper=30
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Saving dataset/models/LightGBMClassifier/trial_0_model.pkl
Finished Task with config: {'feature_fraction': 1.0, 'learning_rate': 0.1, 'min_data_in_leaf': 20, 'num_leaves': 36} and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00learning_rateq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00learning_rateq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3908
 40%|████      | 2/5 [00:19<00:29,  9.94s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.892327203239597, 'learning_rate': 0.022459392534284993, 'min_data_in_leaf': 4, 'num_leaves': 49} and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\x8d\xf1\xc7g\xd8\xccX\r\x00\x00\x00learning_rateq\x02G?\x96\xff\x98Q\xb0\xedfX\x10\x00\x00\x00min_data_in_leafq\x03K\x04X\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\x8d\xf1\xc7g\xd8\xccX\r\x00\x00\x00learning_rateq\x02G?\x96\xff\x98Q\xb0\xedfX\x10\x00\x00\x00min_data_in_leafq\x03K\x04X\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.3922
 60%|██████    | 3/5 [00:45<00:29, 14.68s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8119957623074852, 'learning_rate': 0.073107653039502, 'min_data_in_leaf': 20, 'num_leaves': 33} and reward: 0.3914
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xfb\xde\x89s=XX\r\x00\x00\x00learning_rateq\x02G?\xb2\xb7.\xe2\xe4X\x8bX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K!u.' and reward: 0.3914
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xfb\xde\x89s=XX\r\x00\x00\x00learning_rateq\x02G?\xb2\xb7.\xe2\xe4X\x8bX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K!u.' and reward: 0.3914
 80%|████████  | 4/5 [01:04<00:15, 15.93s/it] 80%|████████  | 4/5 [01:04<00:16, 16.11s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7804982035528809, 'learning_rate': 0.027062887414774417, 'min_data_in_leaf': 14, 'num_leaves': 30} and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xf9\xd7^[\x15\x80X\r\x00\x00\x00learning_rateq\x02G?\x9b\xb6_\xa1\x86\xf2\xf0X\x10\x00\x00\x00min_data_in_leafq\x03K\x0eX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xf9\xd7^[\x15\x80X\r\x00\x00\x00learning_rateq\x02G?\x9b\xb6_\xa1\x86\xf2\xf0X\x10\x00\x00\x00min_data_in_leafq\x03K\x0eX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3912
Time for Gradient Boosting hyperparameter optimization: 82.67767763137817
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.892327203239597, 'learning_rate': 0.022459392534284993, 'min_data_in_leaf': 4, 'num_leaves': 49}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:47<01:11, 23.81s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.47815556313318003, 'embedding_size_factor': 1.4632647476224783, 'layers.choice': 3, 'learning_rate': 0.0006674072463571066, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 5.095645372114737e-11} and reward: 0.3734
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xde\x9a\x19\xca\x83\xad\x9dX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7i\x88K\xc6\xda\x10X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?E\xde\x9e%\xeaq\xdfX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xcc\x03{\xbaYR\x81u.' and reward: 0.3734
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xde\x9a\x19\xca\x83\xad\x9dX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7i\x88K\xc6\xda\x10X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?E\xde\x9e%\xeaq\xdfX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xcc\x03{\xbaYR\x81u.' and reward: 0.3734
 60%|██████    | 3/5 [01:40<01:05, 32.60s/it] 60%|██████    | 3/5 [01:40<01:07, 33.57s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.42160737004445187, 'embedding_size_factor': 0.7036604471843277, 'layers.choice': 2, 'learning_rate': 0.00899697195153318, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 3.2166914363274204e-07} and reward: 0.3828
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xfb\x9dz\x85\xfb\xc1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\x84b\xea\x04\xa7NX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x82m\x01"[\xe1\xd1X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x95\x96;\xda\xd9\xaa\x0bu.' and reward: 0.3828
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xfb\x9dz\x85\xfb\xc1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\x84b\xea\x04\xa7NX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x82m\x01"[\xe1\xd1X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x95\x96;\xda\xd9\xaa\x0bu.' and reward: 0.3828
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 149.7057363986969
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.8s of the -116.2s of remaining time.
Ensemble size: 64
Ensemble weights: 
[0.71875  0.       0.265625 0.       0.015625 0.       0.      ]
	0.394	 = Validation accuracy score
	1.5s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 237.75s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f4a3250a2e8>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master d23c824] ml_store
 1 file changed, 202 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 689020a...d23c824 master -> master (forced update)





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
    test(data_path = "model_fb/fbprophet.json", choice="json" )
TypeError: test() got an unexpected keyword argument 'choice'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master e432985] ml_store
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   d23c824..e432985  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model_old.py 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 
INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:numexpr.utils:NumExpr defaulting to 2 threads.
INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
100%|██████████| 10/10 [00:02<00:00,  3.99it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 2.512 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.238314
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.238313961029053 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa915d7d3c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa915d7d3c8>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]WARNING:root:multiple 5 does not divide base seasonality 1.Falling back to seasonality 1
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 118.99it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1038.6539713541667,
    "abs_error": 365.51055908203125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4218546238360896,
    "sMAPE": 0.5080789476648787,
    "MSIS": 96.87417524746239,
    "QuantileLoss[0.5]": 365.51051330566406,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.22815494802901,
    "NRMSE": 0.6784874725900844,
    "ND": 0.6412465948807565,
    "wQuantileLoss[0.5]": 0.6412465145713404,
    "mean_wQuantileLoss": 0.6412465145713404,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'deepfactor', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  8.01it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.249 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa9194c3208>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa9194c3208>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 112.37it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2262.8567708333335,
    "abs_error": 552.1011962890625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.6581948231980244,
    "sMAPE": 1.8700508550675963,
    "MSIS": 146.3277993985751,
    "QuantileLoss[0.5]": 552.1012096405029,
    "Coverage[0.5]": 0.0,
    "RMSE": 47.5694941200065,
    "NRMSE": 1.0014630341054,
    "ND": 0.9685985899808114,
    "wQuantileLoss[0.5]": 0.9685986134043911,
    "mean_wQuantileLoss": 0.9685986134043911,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'transformer', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:01<00:00,  5.57it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 1.796 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.231945
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.23194522857666 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8e9787b38>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8e9787b38>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 165.17it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 258.6870524088542,
    "abs_error": 176.21302795410156,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1675786811811941,
    "sMAPE": 0.2917166684774411,
    "MSIS": 46.70314967374306,
    "QuantileLoss[0.5]": 176.21304321289062,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.08375119208371,
    "NRMSE": 0.3386052882543939,
    "ND": 0.3091456630773712,
    "wQuantileLoss[0.5]": 0.30914568984717655,
    "mean_wQuantileLoss": 0.30914568984717655,
    "MAE_Coverage": 0.25
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'wavenet', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024
INFO:root:using training windows of length = 12
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in WaveNet: 97636
 30%|███       | 3/10 [00:12<00:28,  4.09s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:24<00:16,  4.04s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:34<00:03,  3.89s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:38<00:00,  3.81s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 38.127 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.859157
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.859156894683838 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8e982af28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8e982af28>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 176.40it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 52950.885416666664,
    "abs_error": 2706.546875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.933443784251292,
    "sMAPE": 1.410310722906468,
    "MSIS": 717.3376478395859,
    "QuantileLoss[0.5]": 2706.5467987060547,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.11059388186948,
    "NRMSE": 4.844433555407779,
    "ND": 4.748327850877193,
    "wQuantileLoss[0.5]": 4.748327717028166,
    "mean_wQuantileLoss": 4.748327717028166,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 59.14it/s, avg_epoch_loss=5.16]
INFO:root:Epoch[0] Elapsed time 0.170 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.162183
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.1621825218200685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8d66ac358>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8d66ac358>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 184.21it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 504.6337076822917,
    "abs_error": 188.70338439941406,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.250339156244931,
    "sMAPE": 0.3177862235444781,
    "MSIS": 50.0135719116196,
    "QuantileLoss[0.5]": 188.70339965820312,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.464053678761804,
    "NRMSE": 0.4729274458686696,
    "ND": 0.33105856912177906,
    "wQuantileLoss[0.5]": 0.33105859589158443,
    "mean_wQuantileLoss": 0.33105859589158443,
    "MAE_Coverage": 0.16666666666666663
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:01<00:00,  9.47it/s, avg_epoch_loss=160]
INFO:root:Epoch[0] Elapsed time 1.057 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=160.137221
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 160.13722096982082 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8d6782320>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8d6782320>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 175.05it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 721.6386473701223,
    "abs_error": 263.04314901123786,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.7429107063314655,
    "sMAPE": 0.5744170818242088,
    "MSIS": 69.71642825325863,
    "QuantileLoss[0.5]": 263.04314901123786,
    "Coverage[0.5]": 0.08333333333333333,
    "RMSE": 26.86333276736381,
    "NRMSE": 0.5655438477339749,
    "ND": 0.4614792087916454,
    "wQuantileLoss[0.5]": 0.4614792087916454,
    "mean_wQuantileLoss": 0.4614792087916454,
    "MAE_Coverage": 0.4166666666666667
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'deepstate', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [02:12<19:51, 132.36s/it, avg_epoch_loss=0.653] 20%|██        | 2/10 [05:21<19:55, 149.46s/it, avg_epoch_loss=0.635] 30%|███       | 3/10 [08:30<18:48, 161.28s/it, avg_epoch_loss=0.618] 40%|████      | 4/10 [11:49<17:15, 172.60s/it, avg_epoch_loss=0.6]   50%|█████     | 5/10 [15:08<15:02, 180.41s/it, avg_epoch_loss=0.582] 60%|██████    | 6/10 [18:24<12:20, 185.12s/it, avg_epoch_loss=0.565] 70%|███████   | 7/10 [22:31<10:11, 203.76s/it, avg_epoch_loss=0.548] 80%|████████  | 8/10 [26:10<06:56, 208.17s/it, avg_epoch_loss=0.531] 90%|█████████ | 9/10 [29:54<03:33, 213.20s/it, avg_epoch_loss=0.516]100%|██████████| 10/10 [33:43<00:00, 217.75s/it, avg_epoch_loss=0.502]100%|██████████| 10/10 [33:43<00:00, 202.34s/it, avg_epoch_loss=0.502]
INFO:root:Epoch[0] Elapsed time 2023.422 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.501657
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5016570329666138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8eb2e1908>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7fa8eb2e1908>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 12.19it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 143.64437866210938,
    "abs_error": 113.09156799316406,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7493390547977266,
    "sMAPE": 0.20097564209453803,
    "MSIS": 29.97356785373142,
    "QuantileLoss[0.5]": 113.0915756225586,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 11.985173284609171,
    "NRMSE": 0.2523194375707194,
    "ND": 0.19840625963712993,
    "wQuantileLoss[0.5]": 0.19840627302203262,
    "mean_wQuantileLoss": 0.19840627302203262,
    "MAE_Coverage": 0.08333333333333331
}

  #### Plot   ####################################################### 


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 9a4f0e7] ml_store
 1 file changed, 498 insertions(+)
To github.com:arita37/mlmodels_store.git
 + db779e7...9a4f0e7 master -> master (forced update)





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model.py 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 4180607] ml_store
 1 file changed, 45 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   9a4f0e7..4180607  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_sklearn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 

  #### metrics   ##################################################### 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f3dfbf67550> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
None

  #### Get  metrics   ################################################ 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
None

  ############ Save/ Load ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master c33c3ad] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   4180607..c33c3ad  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f3a33273e48>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f3a4d5f05f8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.85771953  0.09811225 -0.26046606  1.06032751 -1.39003042 -1.71116766
   0.2656424   1.65712464  1.41767401  0.44509671]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]]
None

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
[[ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]]
None

  ############ Save/ Load ############################################ 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 2de214a] ml_store
 1 file changed, 270 insertions(+)
To github.com:arita37/mlmodels_store.git
   c33c3ad..2de214a  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'dataset/vision/MNIST/', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic:get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/MNIST/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9f565a3ae8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9f565a3ae8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9f565a3ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 18%|█▊        | 1826816/9912422 [00:00<00:00, 17690228.30it/s]9920512it [00:00, 35520747.64it/s]                             
0it [00:00, ?it/s]32768it [00:00, 351397.53it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:13, 124619.36it/s]1654784it [00:00, 9005152.36it/s]                          
0it [00:00, ?it/s]8192it [00:00, 158283.64it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9f565a3ae8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9f565a3ae8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9f565a3ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 431, in <module>
    test(data_path="dataset/json/refactor/resnet18_benchmark_mnist.json", pars_choice="json", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 362, in test
    model, session = fit(model, data_pars, compute_pars, out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 231, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 50, in _train
    image, target = image.to(device), target.to(device)
AttributeError: 'tuple' object has no attribute 'to'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 4910401] ml_store
 1 file changed, 101 insertions(+)
To github.com:arita37/mlmodels_store.git
   2de214a..4910401  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  ############ Dataloader setup  ############################# 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 488, in <module>
    test(pars_choice="json", data_path="model_tch/transformer_sentence.json", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 448, in test
    model, session = fit(model, data_pars, model_pars, compute_pars, out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 139, in fit
    train_data       = SentencesDataset(train_reader.get_examples(train_fname),  model=model.model)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sentence_transformers/readers/NLIDataReader.py", line 21, in get_examples
    mode="rt", encoding="utf-8").readlines()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 53, in open
    binary_file = GzipFile(filename, gz_mode, compresslevel)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 163, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/AllNLI/s1.train.gz'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 37e4d7a] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   4910401..37e4d7a  master -> master





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py 

  Json file path:  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 

  #### Loading params   ############################################## 

  {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2} {'data_info': {'data_path': 'dataset/recommender/', 'dataset': 'IMDB_sample.txt', 'data_type': 'csv_dataset', 'batch_size': 64, 'train': True}, 'preprocessors': [{'uri': 'mlmodels.model_tch.textcnn:split_train_valid', 'args': {'frac': 0.99}}, {'uri': 'mlmodels.model_tch.textcnn:create_tabular_dataset', 'args': {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'}}]} {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f08d6a311e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f08d6a311e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f08d6a311e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f08d6a312f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f08d6a312f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f08d6a312f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=28390905549725e21bd9d610e2d595c806688082af9e00abfdda3fe9ac70b8eb
  Stored in directory: /tmp/pip-ephem-wheel-cache-1711nf_z/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:32:37, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:12:51, 18.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:18:13, 25.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:31:17, 36.7kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:33:14, 52.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.53M/862M [00:01<3:10:03, 74.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<2:12:15, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.5M/862M [00:01<1:32:12, 152kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:01<1:04:22, 217kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.0M/862M [00:01<44:53, 310kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<31:22, 441kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.6M/862M [00:01<21:56, 627kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.0M/862M [00:02<15:22, 890kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<10:48, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.6M/862M [00:02<07:36, 1.78MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<06:09, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:12, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<07:58, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:21, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<04:38, 2.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<07:49, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<07:03, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<05:20, 2.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<06:28, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<07:22, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:44, 2.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:09<04:10, 3.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<10:18, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<08:34, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<06:20, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<07:28, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<06:34, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:12<04:55, 2.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<06:32, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<07:13, 1.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<05:43, 2.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<06:06, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<05:35, 2.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<04:15, 3.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:00, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:31, 2.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<04:11, 3.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<05:58, 2.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:47, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<05:26, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<05:51, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<05:24, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<04:06, 3.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<05:51, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<05:23, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:24<04:05, 3.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:51, 2.16MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:23, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:21, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:04, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:19, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<03:58, 3.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<03:30, 3.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<9:45:05, 21.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<6:49:47, 30.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<4:46:02, 43.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<3:41:45, 55.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<2:37:42, 78.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:50:53, 112kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<1:17:29, 159kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:06:38, 185kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<47:58, 257kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<33:49, 364kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<26:25, 464kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<19:45, 620kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<14:07, 866kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:41, 961kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:07, 1.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<07:22, 1.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:01, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:50, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:05, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:24, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:55, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:21, 2.25MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<03:53, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:41, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:01, 1.49MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:54, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:55, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:03, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:31, 2.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:58, 1.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:35, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:08, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:42, 3.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<41:12, 286kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<30:02, 392kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<21:14, 553kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<17:34, 666kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<14:40, 798kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:45, 1.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<07:40, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<10:43, 1.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:42, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:22, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:08, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:20, 1.58MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:43, 2.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:51, 1.96MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:15, 2.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:58, 2.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:29, 2.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:09, 1.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<04:49, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:29, 3.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<12:28, 912kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:54, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<07:09, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:39, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:38, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:50, 1.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:13, 2.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<09:09, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:32, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:33, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:30, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:48, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:14, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<03:47, 2.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<10:52, 1.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:44, 1.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<06:21, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:01, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:03, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:29, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:36, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:03, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:48, 2.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:13, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:57, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:38, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:22, 3.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:42, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:54, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<05:45, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:33, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:45, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:16, 2.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:09, 2.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<7:59:13, 22.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<5:35:35, 31.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<3:53:54, 45.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<3:08:06, 56.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<2:13:46, 79.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:34:03, 113kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<1:07:14, 157kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<48:08, 219kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<33:51, 311kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<26:02, 402kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<20:20, 515kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<14:45, 709kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<11:58, 870kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:25, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:50, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:12, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:05, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:30, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:35, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:59, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:42, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:56, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:37, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:24, 2.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:10, 3.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<12:21, 820kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<09:41, 1.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:59, 1.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<07:16, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:53, 1.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:22, 2.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:25, 1.84MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:48, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:34, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:51, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:25, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:14, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:03, 3.22MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<14:16, 691kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<10:58, 897kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:55, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<07:49, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<06:27, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:45, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:37, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:54, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:37, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:19, 2.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<9:21:53, 17.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<6:34:04, 24.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<4:35:26, 34.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<3:14:10, 49.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<2:21:43, 67.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<1:40:40, 95.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<1:10:42, 135kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<49:24, 193kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<3:21:46, 47.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<2:24:17, 66.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<1:41:38, 93.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<1:11:04, 133kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<52:42, 179kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<39:35, 239kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<28:22, 333kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<19:56, 472kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<17:13, 545kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<14:35, 642kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<10:47, 868kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<07:41, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:12, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:15, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:17, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<04:33, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:44, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:23, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:42, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:10, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:40, 1.37MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<05:15, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:48, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:19, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:53, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:20, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<03:53, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:16, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:44, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:16, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:50, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:25, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:40, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:07, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<03:42, 2.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:40, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:30, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:04, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:41, 2.40MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:11, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:46, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:29, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<03:15, 2.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:45, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:09, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:44, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<03:26, 2.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:32, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:59, 1.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:37, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<03:21, 2.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:19, 1.62MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:43, 1.51MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:29, 1.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<03:15, 2.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:30, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:32, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:04, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<03:41, 2.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:34, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:34, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:04, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<03:41, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:33, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:51, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:22, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<03:53, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:50, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:20, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:00, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:37, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:37, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<06:03, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:47, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<03:28, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:51, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:54, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:19, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:26, 1.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:43, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:15, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<03:48, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:33, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<5:36:26, 24.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<3:56:23, 34.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<2:45:17, 48.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<1:56:42, 68.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<1:25:34, 93.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<1:00:46, 131kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<42:37, 187kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<31:17, 253kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<23:43, 334kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<16:58, 466kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<11:58, 659kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:54, 720kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<09:38, 814kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:15, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<05:10, 1.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:51, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:47, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:15, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:48, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:45, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:06, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:48, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:29, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:23, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:27, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:10, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:04, 2.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:13, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:43, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:40, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:40, 2.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:34, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:13, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:04, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:58, 2.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:14, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:52, 1.53MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:49, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:46, 2.67MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:39, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:03, 1.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:59, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:54, 2.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:18, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:24, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:08, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<03:00, 2.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:45, 1.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:54, 1.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:35, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<03:18, 2.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:17, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:38, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:21, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:10, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:00, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:22, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:28, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:30, 2.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:23, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:19, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:04, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<02:56, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:53, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:43, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:23, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:09, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:34, 1.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:07, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:39, 1.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<03:20, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<06:05, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:46, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:24, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<03:10, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<06:44, 1.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:26, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:52, 1.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<05:30, 1.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:28, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:14, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<05:13, 1.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:15, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:01, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:54, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<04:12, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:36, 1.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:38, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:37, 2.47MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:45, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:38, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:34, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:34, 2.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<09:37, 665kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<08:17, 772kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:07, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:20, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<06:01, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:29, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:06, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:56, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<05:20, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:56, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:46, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:41, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<08:47, 704kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<07:21, 841kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:26, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<03:51, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<52:43, 116kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<38:02, 161kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<26:50, 227kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<18:43, 323kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<34:47, 174kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<25:44, 235kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<18:21, 329kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<12:50, 467kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<13:00, 460kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<10:29, 570kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<07:40, 778kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<05:25, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<08:05, 731kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<07:01, 841kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<05:13, 1.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:42, 1.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:42, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<05:20, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:04, 1.43MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:55, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<06:14, 926kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<05:42, 1.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:15, 1.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:03, 1.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:17, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:12, 1.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:13, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:18, 2.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:21, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:21, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:22, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:25, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:41, 977kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<05:16, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:57, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:51, 1.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:34, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:00, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:25, 2.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:45, 3.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:07, 887kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:35, 969kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:11, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<02:58, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:31, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:24, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:23, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<05:29, 963kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<05:00, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<03:45, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<02:41, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:50, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:50, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:56, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<02:06, 2.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:50, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:27, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:34, 1.99MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<01:51, 2.74MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<09:15, 549kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<08:43, 582kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<06:38, 764kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<04:45, 1.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:23, 1.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:46, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:46, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:52, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:51, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:05, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:16, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:47, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:49, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:09, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:33, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:06, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:44, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:47, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<02:00, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:37, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:00, 945kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:56, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:50, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:07, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:03, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:20, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:23, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:32, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:56, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<01:24, 3.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:51, 2.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<3:23:41, 22.3kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<2:22:46, 31.7kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<1:39:23, 45.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<1:10:14, 63.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<50:49, 87.8kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<35:56, 124kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<25:05, 176kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<18:34, 237kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<13:17, 330kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<09:20, 467kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<07:36, 569kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<06:07, 706kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:28, 964kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:48, 1.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:25, 1.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:35, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:29, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:30, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:02, 2.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:58, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:24, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:46, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:13, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:17, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:46, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:54, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:03, 1.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:36, 2.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:46, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:58, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:33, 2.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:43, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:55, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:52, 2.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:29, 2.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:38, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:50, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:27, 2.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:37, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:36, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:14, 2.92MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:33, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:19, 1.53MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:56, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:25, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:54, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:56, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:30, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:37, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:43, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:20, 2.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<00:57, 3.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<05:55, 568kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<04:43, 711kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:25, 978kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:25, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:58, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:38, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:59, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:55, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:55, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:27, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:03, 3.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:37, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:23, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:46, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:16, 2.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:32, 872kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:01, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:13, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:34, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:42, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:24, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:47, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:16, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:44, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:25, 1.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:49, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:44, 1.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:43, 1.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:18, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<00:55, 3.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<10:27, 269kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<07:46, 361kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<05:32, 505kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:16, 641kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:27, 791kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:32, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:11, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:00, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:30, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:28, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:00, 1.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:38, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:11, 2.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:28, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:28, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:07, 2.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<00:47, 3.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<21:26, 115kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<15:25, 160kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<10:50, 226kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<07:32, 318kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<1:55:10, 20.8kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<1:20:35, 29.7kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<55:44, 42.4kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<39:08, 59.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<28:11, 82.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<19:51, 117kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<13:46, 166kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<10:08, 223kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<07:29, 301kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<05:17, 424kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<03:39, 601kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:37, 474kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:59, 548kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:59, 731kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<02:06, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:00, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:46, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:19, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:14, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:56, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:40, 2.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:16, 872kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:17, 868kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:46, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:15, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:22, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:17, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:58, 1.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:59, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:00, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:46, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:50, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:53, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:41, 2.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:45, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:07, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:54, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:40, 2.47MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:46, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:49, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:38, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:42, 2.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:46, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:36, 2.59MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:39, 2.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:44, 2.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:34, 2.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:38, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:56, 1.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:47, 1.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:34, 2.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:44, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:45, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:24, 3.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:08, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:01, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:45, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:32, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:46, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:45, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:34, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:35, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:48, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:28, 2.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:35, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:26, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:18, 3.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<20:25, 50.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<14:25, 70.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<09:59, 101kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<06:48, 140kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<04:54, 194kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<03:25, 273kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:25, 366kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:50, 479kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:18, 667kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:03<00:52, 943kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<04:19, 189kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<03:09, 258kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<02:12, 363kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<01:30, 513kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:17, 578kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:01, 722kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:43, 995kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:29, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:50, 804kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:42, 952kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:30, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:26, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:30, 1.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:24, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:17, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:19, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:19, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:14, 2.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:14, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:14, 1.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:07, 3.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<03:04, 131kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<02:12, 182kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<01:30, 257kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:57, 365kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:51, 394kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:38, 512kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:26, 709kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:16, 970kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<12:12, 22.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<08:19, 31.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<05:09, 44.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<03:10, 63.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<02:16, 87.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<01:32, 123kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:55, 175kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:33, 235kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:23, 317kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:15, 444kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:06, 572kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 716kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 979kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 908/400000 [00:00<00:43, 9078.30it/s]  0%|          | 1741/400000 [00:00<00:45, 8840.03it/s]  1%|          | 2617/400000 [00:00<00:45, 8814.19it/s]  1%|          | 3497/400000 [00:00<00:45, 8808.48it/s]  1%|          | 4405/400000 [00:00<00:44, 8887.81it/s]  1%|▏         | 5356/400000 [00:00<00:43, 9065.64it/s]  2%|▏         | 6214/400000 [00:00<00:44, 8914.20it/s]  2%|▏         | 7171/400000 [00:00<00:43, 9098.87it/s]  2%|▏         | 8114/400000 [00:00<00:42, 9192.74it/s]  2%|▏         | 9064/400000 [00:01<00:42, 9281.32it/s]  2%|▏         | 9968/400000 [00:01<00:42, 9183.05it/s]  3%|▎         | 10870/400000 [00:01<00:43, 8942.20it/s]  3%|▎         | 11754/400000 [00:01<00:43, 8828.66it/s]  3%|▎         | 12671/400000 [00:01<00:43, 8927.14it/s]  3%|▎         | 13560/400000 [00:01<00:43, 8902.61it/s]  4%|▎         | 14448/400000 [00:01<00:44, 8634.14it/s]  4%|▍         | 15311/400000 [00:01<00:44, 8614.90it/s]  4%|▍         | 16256/400000 [00:01<00:43, 8849.47it/s]  4%|▍         | 17219/400000 [00:01<00:42, 9069.04it/s]  5%|▍         | 18161/400000 [00:02<00:41, 9171.44it/s]  5%|▍         | 19096/400000 [00:02<00:41, 9219.59it/s]  5%|▌         | 20020/400000 [00:02<00:41, 9113.59it/s]  5%|▌         | 20933/400000 [00:02<00:42, 8911.59it/s]  5%|▌         | 21895/400000 [00:02<00:41, 9091.88it/s]  6%|▌         | 22807/400000 [00:02<00:41, 9051.91it/s]  6%|▌         | 23714/400000 [00:02<00:42, 8765.35it/s]  6%|▌         | 24594/400000 [00:02<00:42, 8745.39it/s]  6%|▋         | 25517/400000 [00:02<00:42, 8883.08it/s]  7%|▋         | 26496/400000 [00:02<00:40, 9135.34it/s]  7%|▋         | 27413/400000 [00:03<00:40, 9145.29it/s]  7%|▋         | 28330/400000 [00:03<00:41, 8927.05it/s]  7%|▋         | 29226/400000 [00:03<00:43, 8595.55it/s]  8%|▊         | 30091/400000 [00:03<00:44, 8276.96it/s]  8%|▊         | 30925/400000 [00:03<00:46, 8013.86it/s]  8%|▊         | 31733/400000 [00:03<00:45, 8017.27it/s]  8%|▊         | 32539/400000 [00:03<00:46, 7935.52it/s]  8%|▊         | 33350/400000 [00:03<00:45, 7985.00it/s]  9%|▊         | 34151/400000 [00:03<00:45, 7967.74it/s]  9%|▊         | 34950/400000 [00:04<00:45, 7957.97it/s]  9%|▉         | 35762/400000 [00:04<00:45, 8005.57it/s]  9%|▉         | 36569/400000 [00:04<00:45, 8021.37it/s]  9%|▉         | 37372/400000 [00:04<00:45, 7976.81it/s] 10%|▉         | 38171/400000 [00:04<00:46, 7851.83it/s] 10%|▉         | 38980/400000 [00:04<00:45, 7921.30it/s] 10%|▉         | 39795/400000 [00:04<00:45, 7987.28it/s] 10%|█         | 40595/400000 [00:04<00:45, 7926.63it/s] 10%|█         | 41398/400000 [00:04<00:45, 7956.84it/s] 11%|█         | 42195/400000 [00:04<00:45, 7917.77it/s] 11%|█         | 43043/400000 [00:05<00:44, 8075.48it/s] 11%|█         | 43852/400000 [00:05<00:44, 7940.54it/s] 11%|█         | 44648/400000 [00:05<00:45, 7882.31it/s] 11%|█▏        | 45493/400000 [00:05<00:44, 8042.44it/s] 12%|█▏        | 46421/400000 [00:05<00:42, 8376.32it/s] 12%|█▏        | 47348/400000 [00:05<00:40, 8623.90it/s] 12%|█▏        | 48290/400000 [00:05<00:39, 8846.96it/s] 12%|█▏        | 49180/400000 [00:05<00:40, 8730.71it/s] 13%|█▎        | 50064/400000 [00:05<00:39, 8761.10it/s] 13%|█▎        | 51005/400000 [00:05<00:39, 8944.00it/s] 13%|█▎        | 51910/400000 [00:06<00:38, 8974.74it/s] 13%|█▎        | 52810/400000 [00:06<00:38, 8960.08it/s] 13%|█▎        | 53708/400000 [00:06<00:39, 8874.63it/s] 14%|█▎        | 54616/400000 [00:06<00:38, 8934.95it/s] 14%|█▍        | 55542/400000 [00:06<00:38, 9027.82it/s] 14%|█▍        | 56446/400000 [00:06<00:38, 8929.35it/s] 14%|█▍        | 57340/400000 [00:06<00:38, 8853.59it/s] 15%|█▍        | 58227/400000 [00:06<00:38, 8846.07it/s] 15%|█▍        | 59142/400000 [00:06<00:38, 8933.69it/s] 15%|█▌        | 60096/400000 [00:06<00:37, 9105.61it/s] 15%|█▌        | 61044/400000 [00:07<00:36, 9213.03it/s] 15%|█▌        | 61967/400000 [00:07<00:37, 9130.28it/s] 16%|█▌        | 62882/400000 [00:07<00:37, 9062.94it/s] 16%|█▌        | 63807/400000 [00:07<00:36, 9117.31it/s] 16%|█▌        | 64720/400000 [00:07<00:37, 8925.78it/s] 16%|█▋        | 65614/400000 [00:07<00:39, 8547.22it/s] 17%|█▋        | 66503/400000 [00:07<00:38, 8644.81it/s] 17%|█▋        | 67392/400000 [00:07<00:38, 8715.47it/s] 17%|█▋        | 68345/400000 [00:07<00:37, 8941.26it/s] 17%|█▋        | 69285/400000 [00:07<00:36, 9071.70it/s] 18%|█▊        | 70195/400000 [00:08<00:36, 9015.98it/s] 18%|█▊        | 71100/400000 [00:08<00:36, 9025.68it/s] 18%|█▊        | 72004/400000 [00:08<00:37, 8744.76it/s] 18%|█▊        | 72914/400000 [00:08<00:36, 8847.43it/s] 18%|█▊        | 73846/400000 [00:08<00:36, 8982.99it/s] 19%|█▊        | 74812/400000 [00:08<00:35, 9174.03it/s] 19%|█▉        | 75793/400000 [00:08<00:34, 9353.65it/s] 19%|█▉        | 76731/400000 [00:08<00:34, 9245.54it/s] 19%|█▉        | 77678/400000 [00:08<00:34, 9310.32it/s] 20%|█▉        | 78641/400000 [00:08<00:34, 9402.47it/s] 20%|█▉        | 79607/400000 [00:09<00:33, 9477.75it/s] 20%|██        | 80556/400000 [00:09<00:34, 9361.53it/s] 20%|██        | 81494/400000 [00:09<00:34, 9170.06it/s] 21%|██        | 82426/400000 [00:09<00:34, 9211.78it/s] 21%|██        | 83349/400000 [00:09<00:34, 9160.72it/s] 21%|██        | 84266/400000 [00:09<00:35, 8941.76it/s] 21%|██▏       | 85171/400000 [00:09<00:35, 8971.55it/s] 22%|██▏       | 86070/400000 [00:09<00:35, 8918.24it/s] 22%|██▏       | 86963/400000 [00:09<00:37, 8255.49it/s] 22%|██▏       | 87817/400000 [00:10<00:37, 8338.80it/s] 22%|██▏       | 88716/400000 [00:10<00:36, 8521.78it/s] 22%|██▏       | 89602/400000 [00:10<00:36, 8620.04it/s] 23%|██▎       | 90469/400000 [00:10<00:36, 8463.70it/s] 23%|██▎       | 91368/400000 [00:10<00:35, 8614.21it/s] 23%|██▎       | 92281/400000 [00:10<00:35, 8760.85it/s] 23%|██▎       | 93168/400000 [00:10<00:34, 8792.80it/s] 24%|██▎       | 94050/400000 [00:10<00:35, 8685.85it/s] 24%|██▎       | 94939/400000 [00:10<00:34, 8745.51it/s] 24%|██▍       | 95899/400000 [00:10<00:33, 8983.87it/s] 24%|██▍       | 96800/400000 [00:11<00:33, 8952.35it/s] 24%|██▍       | 97698/400000 [00:11<00:35, 8591.94it/s] 25%|██▍       | 98562/400000 [00:11<00:35, 8591.02it/s] 25%|██▍       | 99429/400000 [00:11<00:34, 8612.19it/s] 25%|██▌       | 100375/400000 [00:11<00:33, 8848.02it/s] 25%|██▌       | 101263/400000 [00:11<00:33, 8827.44it/s] 26%|██▌       | 102148/400000 [00:11<00:33, 8764.61it/s] 26%|██▌       | 103071/400000 [00:11<00:33, 8897.80it/s] 26%|██▌       | 103963/400000 [00:11<00:33, 8763.92it/s] 26%|██▌       | 104902/400000 [00:11<00:33, 8941.72it/s] 26%|██▋       | 105829/400000 [00:12<00:32, 9036.03it/s] 27%|██▋       | 106763/400000 [00:12<00:32, 9124.55it/s] 27%|██▋       | 107707/400000 [00:12<00:31, 9214.23it/s] 27%|██▋       | 108630/400000 [00:12<00:32, 9087.73it/s] 27%|██▋       | 109540/400000 [00:12<00:34, 8534.63it/s] 28%|██▊       | 110402/400000 [00:12<00:34, 8496.35it/s] 28%|██▊       | 111270/400000 [00:12<00:33, 8548.73it/s] 28%|██▊       | 112152/400000 [00:12<00:33, 8627.70it/s] 28%|██▊       | 113018/400000 [00:12<00:34, 8421.18it/s] 28%|██▊       | 113864/400000 [00:13<00:34, 8246.37it/s] 29%|██▊       | 114748/400000 [00:13<00:33, 8414.45it/s] 29%|██▉       | 115656/400000 [00:13<00:33, 8601.86it/s] 29%|██▉       | 116520/400000 [00:13<00:33, 8405.47it/s] 29%|██▉       | 117432/400000 [00:13<00:32, 8607.57it/s] 30%|██▉       | 118386/400000 [00:13<00:31, 8866.49it/s] 30%|██▉       | 119311/400000 [00:13<00:31, 8976.03it/s] 30%|███       | 120212/400000 [00:13<00:31, 8748.01it/s] 30%|███       | 121103/400000 [00:13<00:31, 8789.08it/s] 31%|███       | 122028/400000 [00:13<00:31, 8920.36it/s] 31%|███       | 122973/400000 [00:14<00:30, 9072.87it/s] 31%|███       | 123926/400000 [00:14<00:29, 9204.18it/s] 31%|███       | 124877/400000 [00:14<00:29, 9292.70it/s] 31%|███▏      | 125808/400000 [00:14<00:30, 9119.98it/s] 32%|███▏      | 126722/400000 [00:14<00:30, 9021.50it/s] 32%|███▏      | 127626/400000 [00:14<00:30, 8944.28it/s] 32%|███▏      | 128534/400000 [00:14<00:30, 8981.48it/s] 32%|███▏      | 129434/400000 [00:14<00:30, 8812.91it/s] 33%|███▎      | 130318/400000 [00:14<00:30, 8817.92it/s] 33%|███▎      | 131201/400000 [00:14<00:31, 8616.08it/s] 33%|███▎      | 132070/400000 [00:15<00:31, 8637.41it/s] 33%|███▎      | 132960/400000 [00:15<00:30, 8712.16it/s] 33%|███▎      | 133833/400000 [00:15<00:30, 8657.10it/s] 34%|███▎      | 134700/400000 [00:15<00:31, 8522.44it/s] 34%|███▍      | 135596/400000 [00:15<00:30, 8647.28it/s] 34%|███▍      | 136495/400000 [00:15<00:30, 8744.07it/s] 34%|███▍      | 137427/400000 [00:15<00:29, 8908.17it/s] 35%|███▍      | 138320/400000 [00:15<00:29, 8887.09it/s] 35%|███▍      | 139210/400000 [00:15<00:29, 8736.12it/s] 35%|███▌      | 140137/400000 [00:15<00:29, 8886.71it/s] 35%|███▌      | 141102/400000 [00:16<00:28, 9101.29it/s] 36%|███▌      | 142015/400000 [00:16<00:28, 9032.99it/s] 36%|███▌      | 142935/400000 [00:16<00:28, 9082.10it/s] 36%|███▌      | 143845/400000 [00:16<00:28, 9017.60it/s] 36%|███▌      | 144748/400000 [00:16<00:28, 9015.29it/s] 36%|███▋      | 145651/400000 [00:16<00:28, 8935.09it/s] 37%|███▋      | 146546/400000 [00:16<00:28, 8800.77it/s] 37%|███▋      | 147432/400000 [00:16<00:28, 8816.56it/s] 37%|███▋      | 148315/400000 [00:16<00:28, 8763.29it/s] 37%|███▋      | 149252/400000 [00:17<00:28, 8935.08it/s] 38%|███▊      | 150184/400000 [00:17<00:27, 9046.61it/s] 38%|███▊      | 151090/400000 [00:17<00:27, 9007.43it/s] 38%|███▊      | 152005/400000 [00:17<00:27, 9049.38it/s] 38%|███▊      | 152911/400000 [00:17<00:27, 8899.72it/s] 38%|███▊      | 153856/400000 [00:17<00:27, 9056.16it/s] 39%|███▊      | 154763/400000 [00:17<00:27, 8918.35it/s] 39%|███▉      | 155737/400000 [00:17<00:26, 9149.91it/s] 39%|███▉      | 156717/400000 [00:17<00:26, 9334.00it/s] 39%|███▉      | 157654/400000 [00:17<00:26, 9205.84it/s] 40%|███▉      | 158582/400000 [00:18<00:26, 9227.45it/s] 40%|███▉      | 159507/400000 [00:18<00:26, 9213.49it/s] 40%|████      | 160430/400000 [00:18<00:26, 9202.95it/s] 40%|████      | 161352/400000 [00:18<00:26, 8945.15it/s] 41%|████      | 162249/400000 [00:18<00:26, 8844.56it/s] 41%|████      | 163217/400000 [00:18<00:26, 9078.09it/s] 41%|████      | 164128/400000 [00:18<00:26, 9005.99it/s] 41%|████▏     | 165032/400000 [00:18<00:26, 9014.57it/s] 41%|████▏     | 165935/400000 [00:18<00:26, 8997.08it/s] 42%|████▏     | 166836/400000 [00:18<00:26, 8954.67it/s] 42%|████▏     | 167791/400000 [00:19<00:25, 9123.77it/s] 42%|████▏     | 168724/400000 [00:19<00:25, 9183.61it/s] 42%|████▏     | 169644/400000 [00:19<00:25, 9094.54it/s] 43%|████▎     | 170590/400000 [00:19<00:24, 9200.14it/s] 43%|████▎     | 171511/400000 [00:19<00:25, 9104.13it/s] 43%|████▎     | 172465/400000 [00:19<00:24, 9229.18it/s] 43%|████▎     | 173403/400000 [00:19<00:24, 9273.06it/s] 44%|████▎     | 174332/400000 [00:19<00:24, 9227.93it/s] 44%|████▍     | 175256/400000 [00:19<00:24, 9171.50it/s] 44%|████▍     | 176174/400000 [00:19<00:25, 8663.79it/s] 44%|████▍     | 177072/400000 [00:20<00:25, 8754.08it/s] 44%|████▍     | 177978/400000 [00:20<00:25, 8843.09it/s] 45%|████▍     | 178891/400000 [00:20<00:24, 8924.74it/s] 45%|████▍     | 179816/400000 [00:20<00:24, 9018.65it/s] 45%|████▌     | 180720/400000 [00:20<00:24, 8934.81it/s] 45%|████▌     | 181675/400000 [00:20<00:23, 9108.41it/s] 46%|████▌     | 182642/400000 [00:20<00:23, 9268.28it/s] 46%|████▌     | 183580/400000 [00:20<00:23, 9300.03it/s] 46%|████▌     | 184557/400000 [00:20<00:22, 9433.89it/s] 46%|████▋     | 185502/400000 [00:20<00:23, 9262.37it/s] 47%|████▋     | 186467/400000 [00:21<00:22, 9373.18it/s] 47%|████▋     | 187406/400000 [00:21<00:23, 9215.07it/s] 47%|████▋     | 188330/400000 [00:21<00:23, 9137.62it/s] 47%|████▋     | 189261/400000 [00:21<00:22, 9188.61it/s] 48%|████▊     | 190181/400000 [00:21<00:23, 8946.90it/s] 48%|████▊     | 191078/400000 [00:21<00:23, 8785.78it/s] 48%|████▊     | 191959/400000 [00:21<00:23, 8772.57it/s] 48%|████▊     | 192838/400000 [00:21<00:23, 8767.10it/s] 48%|████▊     | 193725/400000 [00:21<00:23, 8795.56it/s] 49%|████▊     | 194606/400000 [00:22<00:23, 8795.21it/s] 49%|████▉     | 195506/400000 [00:22<00:23, 8853.38it/s] 49%|████▉     | 196456/400000 [00:22<00:22, 9035.82it/s] 49%|████▉     | 197361/400000 [00:22<00:22, 8866.18it/s] 50%|████▉     | 198259/400000 [00:22<00:22, 8898.75it/s] 50%|████▉     | 199151/400000 [00:22<00:22, 8850.28it/s] 50%|█████     | 200058/400000 [00:22<00:22, 8915.02it/s] 50%|█████     | 201021/400000 [00:22<00:21, 9116.84it/s] 50%|█████     | 201989/400000 [00:22<00:21, 9277.76it/s] 51%|█████     | 202919/400000 [00:22<00:21, 9198.63it/s] 51%|█████     | 203841/400000 [00:23<00:22, 8759.92it/s] 51%|█████     | 204723/400000 [00:23<00:22, 8594.68it/s] 51%|█████▏    | 205587/400000 [00:23<00:22, 8581.32it/s] 52%|█████▏    | 206449/400000 [00:23<00:22, 8500.56it/s] 52%|█████▏    | 207302/400000 [00:23<00:22, 8478.26it/s] 52%|█████▏    | 208152/400000 [00:23<00:22, 8479.17it/s] 52%|█████▏    | 209096/400000 [00:23<00:21, 8744.23it/s] 52%|█████▏    | 209984/400000 [00:23<00:21, 8783.93it/s] 53%|█████▎    | 210886/400000 [00:23<00:21, 8852.68it/s] 53%|█████▎    | 211777/400000 [00:23<00:21, 8867.67it/s] 53%|█████▎    | 212665/400000 [00:24<00:21, 8678.81it/s] 53%|█████▎    | 213535/400000 [00:24<00:21, 8672.13it/s] 54%|█████▎    | 214436/400000 [00:24<00:21, 8769.02it/s] 54%|█████▍    | 215361/400000 [00:24<00:20, 8907.18it/s] 54%|█████▍    | 216274/400000 [00:24<00:20, 8963.31it/s] 54%|█████▍    | 217201/400000 [00:24<00:20, 9051.77it/s] 55%|█████▍    | 218108/400000 [00:24<00:20, 8969.69it/s] 55%|█████▍    | 219006/400000 [00:24<00:20, 8920.79it/s] 55%|█████▍    | 219901/400000 [00:24<00:20, 8928.57it/s] 55%|█████▌    | 220795/400000 [00:24<00:20, 8605.53it/s] 55%|█████▌    | 221659/400000 [00:25<00:20, 8594.74it/s] 56%|█████▌    | 222521/400000 [00:25<00:20, 8515.66it/s] 56%|█████▌    | 223375/400000 [00:25<00:20, 8500.82it/s] 56%|█████▌    | 224265/400000 [00:25<00:20, 8616.69it/s] 56%|█████▋    | 225134/400000 [00:25<00:20, 8638.16it/s] 57%|█████▋    | 226082/400000 [00:25<00:19, 8872.30it/s] 57%|█████▋    | 227038/400000 [00:25<00:19, 9066.14it/s] 57%|█████▋    | 227958/400000 [00:25<00:18, 9103.87it/s] 57%|█████▋    | 228933/400000 [00:25<00:18, 9285.98it/s] 57%|█████▋    | 229864/400000 [00:25<00:18, 9279.84it/s] 58%|█████▊    | 230794/400000 [00:26<00:18, 8918.66it/s] 58%|█████▊    | 231690/400000 [00:26<00:18, 8922.25it/s] 58%|█████▊    | 232645/400000 [00:26<00:18, 9101.43it/s] 58%|█████▊    | 233593/400000 [00:26<00:18, 9211.75it/s] 59%|█████▊    | 234517/400000 [00:26<00:18, 9166.36it/s] 59%|█████▉    | 235436/400000 [00:26<00:17, 9163.87it/s] 59%|█████▉    | 236360/400000 [00:26<00:17, 9185.87it/s] 59%|█████▉    | 237301/400000 [00:26<00:17, 9251.93it/s] 60%|█████▉    | 238252/400000 [00:26<00:17, 9326.71it/s] 60%|█████▉    | 239186/400000 [00:26<00:17, 9264.19it/s] 60%|██████    | 240124/400000 [00:27<00:17, 9298.30it/s] 60%|██████    | 241055/400000 [00:27<00:17, 9055.17it/s] 60%|██████    | 241963/400000 [00:27<00:17, 9020.02it/s] 61%|██████    | 242892/400000 [00:27<00:17, 9098.59it/s] 61%|██████    | 243822/400000 [00:27<00:17, 9155.67it/s] 61%|██████    | 244739/400000 [00:27<00:17, 9111.94it/s] 61%|██████▏   | 245694/400000 [00:27<00:16, 9238.65it/s] 62%|██████▏   | 246619/400000 [00:27<00:16, 9227.75it/s] 62%|██████▏   | 247543/400000 [00:27<00:16, 9139.58it/s] 62%|██████▏   | 248479/400000 [00:28<00:16, 9198.98it/s] 62%|██████▏   | 249400/400000 [00:28<00:16, 9190.57it/s] 63%|██████▎   | 250320/400000 [00:28<00:16, 9173.32it/s] 63%|██████▎   | 251238/400000 [00:28<00:16, 8812.40it/s] 63%|██████▎   | 252123/400000 [00:28<00:17, 8649.72it/s] 63%|██████▎   | 253006/400000 [00:28<00:16, 8695.00it/s] 63%|██████▎   | 253882/400000 [00:28<00:16, 8713.45it/s] 64%|██████▎   | 254787/400000 [00:28<00:16, 8810.97it/s] 64%|██████▍   | 255670/400000 [00:28<00:16, 8576.82it/s] 64%|██████▍   | 256530/400000 [00:28<00:16, 8506.16it/s] 64%|██████▍   | 257383/400000 [00:29<00:16, 8410.73it/s] 65%|██████▍   | 258226/400000 [00:29<00:17, 8161.53it/s] 65%|██████▍   | 259125/400000 [00:29<00:16, 8393.11it/s] 65%|██████▌   | 260082/400000 [00:29<00:16, 8714.23it/s] 65%|██████▌   | 261022/400000 [00:29<00:15, 8906.38it/s] 65%|██████▌   | 261918/400000 [00:29<00:15, 8640.61it/s] 66%|██████▌   | 262788/400000 [00:29<00:15, 8601.53it/s] 66%|██████▌   | 263652/400000 [00:29<00:16, 8487.66it/s] 66%|██████▌   | 264504/400000 [00:29<00:16, 8316.59it/s] 66%|██████▋   | 265339/400000 [00:30<00:17, 7908.39it/s] 67%|██████▋   | 266136/400000 [00:30<00:18, 7375.02it/s] 67%|██████▋   | 266886/400000 [00:30<00:17, 7395.42it/s] 67%|██████▋   | 267656/400000 [00:30<00:17, 7481.65it/s] 67%|██████▋   | 268411/400000 [00:30<00:17, 7485.62it/s] 67%|██████▋   | 269164/400000 [00:30<00:17, 7472.01it/s] 67%|██████▋   | 269915/400000 [00:30<00:17, 7470.68it/s] 68%|██████▊   | 270817/400000 [00:30<00:16, 7875.41it/s] 68%|██████▊   | 271739/400000 [00:30<00:15, 8234.33it/s] 68%|██████▊   | 272608/400000 [00:30<00:15, 8364.08it/s] 68%|██████▊   | 273524/400000 [00:31<00:14, 8586.41it/s] 69%|██████▊   | 274390/400000 [00:31<00:14, 8573.53it/s] 69%|██████▉   | 275315/400000 [00:31<00:14, 8762.50it/s] 69%|██████▉   | 276265/400000 [00:31<00:13, 8971.06it/s] 69%|██████▉   | 277167/400000 [00:31<00:13, 8971.80it/s] 70%|██████▉   | 278097/400000 [00:31<00:13, 9066.00it/s] 70%|██████▉   | 279006/400000 [00:31<00:13, 8786.74it/s] 70%|██████▉   | 279940/400000 [00:31<00:13, 8944.75it/s] 70%|███████   | 280878/400000 [00:31<00:13, 9068.96it/s] 70%|███████   | 281836/400000 [00:31<00:12, 9214.71it/s] 71%|███████   | 282805/400000 [00:32<00:12, 9350.46it/s] 71%|███████   | 283743/400000 [00:32<00:12, 9209.45it/s] 71%|███████   | 284666/400000 [00:32<00:12, 9067.51it/s] 71%|███████▏  | 285575/400000 [00:32<00:12, 9027.48it/s] 72%|███████▏  | 286480/400000 [00:32<00:13, 8714.62it/s] 72%|███████▏  | 287355/400000 [00:32<00:13, 8397.77it/s] 72%|███████▏  | 288200/400000 [00:32<00:13, 8012.78it/s] 72%|███████▏  | 289139/400000 [00:32<00:13, 8380.79it/s] 73%|███████▎  | 290080/400000 [00:32<00:12, 8663.69it/s] 73%|███████▎  | 291056/400000 [00:33<00:12, 8964.80it/s] 73%|███████▎  | 291962/400000 [00:33<00:12, 8912.13it/s] 73%|███████▎  | 292860/400000 [00:33<00:12, 8515.06it/s] 73%|███████▎  | 293720/400000 [00:33<00:12, 8442.93it/s] 74%|███████▎  | 294622/400000 [00:33<00:12, 8608.00it/s] 74%|███████▍  | 295582/400000 [00:33<00:11, 8882.70it/s] 74%|███████▍  | 296476/400000 [00:33<00:11, 8849.28it/s] 74%|███████▍  | 297365/400000 [00:33<00:11, 8721.54it/s] 75%|███████▍  | 298241/400000 [00:33<00:12, 8479.60it/s] 75%|███████▍  | 299136/400000 [00:33<00:11, 8614.18it/s] 75%|███████▌  | 300001/400000 [00:34<00:11, 8562.46it/s] 75%|███████▌  | 300864/400000 [00:34<00:11, 8582.30it/s] 75%|███████▌  | 301724/400000 [00:34<00:11, 8401.50it/s] 76%|███████▌  | 302677/400000 [00:34<00:11, 8710.23it/s] 76%|███████▌  | 303611/400000 [00:34<00:10, 8889.79it/s] 76%|███████▌  | 304509/400000 [00:34<00:10, 8914.60it/s] 76%|███████▋  | 305404/400000 [00:34<00:10, 8665.51it/s] 77%|███████▋  | 306293/400000 [00:34<00:10, 8731.16it/s] 77%|███████▋  | 307182/400000 [00:34<00:10, 8778.07it/s] 77%|███████▋  | 308074/400000 [00:34<00:10, 8818.95it/s] 77%|███████▋  | 308958/400000 [00:35<00:10, 8756.13it/s] 77%|███████▋  | 309835/400000 [00:35<00:10, 8455.64it/s] 78%|███████▊  | 310684/400000 [00:35<00:10, 8419.98it/s] 78%|███████▊  | 311529/400000 [00:35<00:10, 8219.46it/s] 78%|███████▊  | 312380/400000 [00:35<00:10, 8302.67it/s] 78%|███████▊  | 313213/400000 [00:35<00:10, 8233.75it/s] 79%|███████▊  | 314052/400000 [00:35<00:10, 8277.93it/s] 79%|███████▊  | 314898/400000 [00:35<00:10, 8330.94it/s] 79%|███████▉  | 315773/400000 [00:35<00:09, 8451.90it/s] 79%|███████▉  | 316683/400000 [00:35<00:09, 8636.13it/s] 79%|███████▉  | 317591/400000 [00:36<00:09, 8763.11it/s] 80%|███████▉  | 318469/400000 [00:36<00:09, 8551.96it/s] 80%|███████▉  | 319350/400000 [00:36<00:09, 8626.35it/s] 80%|████████  | 320264/400000 [00:36<00:09, 8773.68it/s] 80%|████████  | 321144/400000 [00:36<00:09, 8718.72it/s] 81%|████████  | 322018/400000 [00:36<00:08, 8721.69it/s] 81%|████████  | 322892/400000 [00:36<00:08, 8583.39it/s] 81%|████████  | 323752/400000 [00:36<00:09, 8288.32it/s] 81%|████████  | 324584/400000 [00:36<00:09, 8274.43it/s] 81%|████████▏ | 325414/400000 [00:37<00:09, 8188.57it/s] 82%|████████▏ | 326235/400000 [00:37<00:09, 8095.47it/s] 82%|████████▏ | 327046/400000 [00:37<00:09, 7856.23it/s] 82%|████████▏ | 327856/400000 [00:37<00:09, 7927.49it/s] 82%|████████▏ | 328659/400000 [00:37<00:08, 7956.98it/s] 82%|████████▏ | 329480/400000 [00:37<00:08, 8030.65it/s] 83%|████████▎ | 330285/400000 [00:37<00:08, 7952.88it/s] 83%|████████▎ | 331082/400000 [00:37<00:08, 7871.25it/s] 83%|████████▎ | 331870/400000 [00:37<00:08, 7780.36it/s] 83%|████████▎ | 332649/400000 [00:37<00:08, 7704.00it/s] 83%|████████▎ | 333421/400000 [00:38<00:08, 7664.29it/s] 84%|████████▎ | 334189/400000 [00:38<00:08, 7566.87it/s] 84%|████████▎ | 334947/400000 [00:38<00:08, 7430.41it/s] 84%|████████▍ | 335692/400000 [00:38<00:08, 7365.56it/s] 84%|████████▍ | 336430/400000 [00:38<00:08, 7286.15it/s] 84%|████████▍ | 337160/400000 [00:38<00:08, 7256.67it/s] 84%|████████▍ | 337887/400000 [00:38<00:08, 7227.63it/s] 85%|████████▍ | 338611/400000 [00:38<00:08, 7225.59it/s] 85%|████████▍ | 339334/400000 [00:38<00:08, 7086.20it/s] 85%|████████▌ | 340054/400000 [00:38<00:08, 7118.81it/s] 85%|████████▌ | 340782/400000 [00:39<00:08, 7163.82it/s] 85%|████████▌ | 341514/400000 [00:39<00:08, 7208.93it/s] 86%|████████▌ | 342236/400000 [00:39<00:08, 7210.94it/s] 86%|████████▌ | 342958/400000 [00:39<00:07, 7163.18it/s] 86%|████████▌ | 343675/400000 [00:39<00:07, 7133.37it/s] 86%|████████▌ | 344406/400000 [00:39<00:07, 7182.13it/s] 86%|████████▋ | 345125/400000 [00:39<00:07, 7174.66it/s] 86%|████████▋ | 345894/400000 [00:39<00:07, 7321.72it/s] 87%|████████▋ | 346667/400000 [00:39<00:07, 7437.66it/s] 87%|████████▋ | 347417/400000 [00:39<00:07, 7454.10it/s] 87%|████████▋ | 348164/400000 [00:40<00:07, 7378.20it/s] 87%|████████▋ | 348903/400000 [00:40<00:07, 7286.76it/s] 87%|████████▋ | 349633/400000 [00:40<00:06, 7217.75it/s] 88%|████████▊ | 350384/400000 [00:40<00:06, 7302.51it/s] 88%|████████▊ | 351157/400000 [00:40<00:06, 7424.32it/s] 88%|████████▊ | 351901/400000 [00:40<00:06, 7417.13it/s] 88%|████████▊ | 352646/400000 [00:40<00:06, 7426.46it/s] 88%|████████▊ | 353390/400000 [00:40<00:06, 7385.28it/s] 89%|████████▊ | 354135/400000 [00:40<00:06, 7402.97it/s] 89%|████████▊ | 354912/400000 [00:41<00:06, 7508.76it/s] 89%|████████▉ | 355674/400000 [00:41<00:05, 7541.36it/s] 89%|████████▉ | 356429/400000 [00:41<00:05, 7472.63it/s] 89%|████████▉ | 357224/400000 [00:41<00:05, 7606.96it/s] 89%|████████▉ | 357986/400000 [00:41<00:05, 7607.85it/s] 90%|████████▉ | 358748/400000 [00:41<00:05, 7594.01it/s] 90%|████████▉ | 359518/400000 [00:41<00:05, 7623.56it/s] 90%|█████████ | 360283/400000 [00:41<00:05, 7630.74it/s] 90%|█████████ | 361047/400000 [00:41<00:05, 7547.23it/s] 90%|█████████ | 361803/400000 [00:41<00:05, 7526.99it/s] 91%|█████████ | 362566/400000 [00:42<00:04, 7555.43it/s] 91%|█████████ | 363322/400000 [00:42<00:04, 7523.89it/s] 91%|█████████ | 364088/400000 [00:42<00:04, 7561.28it/s] 91%|█████████ | 364845/400000 [00:42<00:04, 7504.12it/s] 91%|█████████▏| 365613/400000 [00:42<00:04, 7554.00it/s] 92%|█████████▏| 366380/400000 [00:42<00:04, 7587.93it/s] 92%|█████████▏| 367168/400000 [00:42<00:04, 7671.48it/s] 92%|█████████▏| 367983/400000 [00:42<00:04, 7808.45it/s] 92%|█████████▏| 368765/400000 [00:42<00:03, 7809.58it/s] 92%|█████████▏| 369599/400000 [00:42<00:03, 7960.23it/s] 93%|█████████▎| 370438/400000 [00:43<00:03, 8084.43it/s] 93%|█████████▎| 371258/400000 [00:43<00:03, 8116.59it/s] 93%|█████████▎| 372071/400000 [00:43<00:03, 7825.78it/s] 93%|█████████▎| 372857/400000 [00:43<00:03, 7713.89it/s] 93%|█████████▎| 373631/400000 [00:43<00:03, 7698.41it/s] 94%|█████████▎| 374416/400000 [00:43<00:03, 7740.89it/s] 94%|█████████▍| 375210/400000 [00:43<00:03, 7798.35it/s] 94%|█████████▍| 376004/400000 [00:43<00:03, 7839.64it/s] 94%|█████████▍| 376789/400000 [00:43<00:02, 7786.63it/s] 94%|█████████▍| 377589/400000 [00:43<00:02, 7847.86it/s] 95%|█████████▍| 378383/400000 [00:44<00:02, 7872.82it/s] 95%|█████████▍| 379204/400000 [00:44<00:02, 7969.05it/s] 95%|█████████▌| 380017/400000 [00:44<00:02, 8016.34it/s] 95%|█████████▌| 380820/400000 [00:44<00:02, 7868.79it/s] 95%|█████████▌| 381614/400000 [00:44<00:02, 7889.07it/s] 96%|█████████▌| 382404/400000 [00:44<00:02, 7877.51it/s] 96%|█████████▌| 383193/400000 [00:44<00:02, 7817.78it/s] 96%|█████████▌| 383976/400000 [00:44<00:02, 7777.39it/s] 96%|█████████▌| 384755/400000 [00:44<00:01, 7712.41it/s] 96%|█████████▋| 385527/400000 [00:44<00:01, 7685.24it/s] 97%|█████████▋| 386309/400000 [00:45<00:01, 7723.06it/s] 97%|█████████▋| 387082/400000 [00:45<00:01, 7646.29it/s] 97%|█████████▋| 387847/400000 [00:45<00:01, 7565.08it/s] 97%|█████████▋| 388604/400000 [00:45<00:01, 7516.98it/s] 97%|█████████▋| 389390/400000 [00:45<00:01, 7616.25it/s] 98%|█████████▊| 390153/400000 [00:45<00:01, 7613.38it/s] 98%|█████████▊| 390957/400000 [00:45<00:01, 7736.45it/s] 98%|█████████▊| 391733/400000 [00:45<00:01, 7742.49it/s] 98%|█████████▊| 392508/400000 [00:45<00:00, 7663.51it/s] 98%|█████████▊| 393275/400000 [00:45<00:00, 7648.79it/s] 99%|█████████▊| 394041/400000 [00:46<00:00, 7629.44it/s] 99%|█████████▊| 394822/400000 [00:46<00:00, 7681.40it/s] 99%|█████████▉| 395591/400000 [00:46<00:00, 7654.83it/s] 99%|█████████▉| 396378/400000 [00:46<00:00, 7718.09it/s] 99%|█████████▉| 397172/400000 [00:46<00:00, 7782.34it/s] 99%|█████████▉| 397951/400000 [00:46<00:00, 7767.19it/s]100%|█████████▉| 398728/400000 [00:46<00:00, 7696.46it/s]100%|█████████▉| 399500/400000 [00:46<00:00, 7701.85it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8538.53it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
3

  #### Model init       ############################################# 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Model fit        ############################################# 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f08d6a311e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f08d6a311e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f08d6a311e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f08d6a312f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f08d6a312f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f08d6a312f0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011518194668206127 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.06813670396804809 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  #### Predict from Load   ########################################### 
{'data_info': {'data_path': 'dataset/recommender/', 'dataset': 'IMDB_sample.txt', 'data_type': 'csv_dataset', 'batch_size': 64, 'train': True}, 'preprocessors': [{'uri': 'mlmodels.model_tch.textcnn:split_train_valid', 'args': {'frac': 0.99}}, {'uri': 'mlmodels.model_tch.textcnn:create_tabular_dataset', 'args': {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'}}], 'train': 0, 'frac': 1}

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f08d6a311e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f08d6a311e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f08d6a311e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f08d6a312f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f08d6a312f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f08d6a312f0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  #### Predict   ##################################################### 
{'data_info': {'data_path': 'dataset/recommender/', 'dataset': 'IMDB_sample.txt', 'data_type': 'csv_dataset', 'batch_size': 64, 'train': True}, 'preprocessors': [{'uri': 'mlmodels.model_tch.textcnn:split_train_valid', 'args': {'frac': 0.99}}, {'uri': 'mlmodels.model_tch.textcnn:create_tabular_dataset', 'args': {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'}}], 'train': 0, 'frac': 1}

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f08d6a311e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f08d6a311e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f08d6a311e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f08d6a312f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f08d6a312f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f08d6a312f0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  #### metrics   ##################################################### 

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f08d6a311e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f08d6a311e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f08d6a311e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f08d6a312f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f08d6a312f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f08d6a312f0> , (data_info, **args) 

  Download en 
Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (2.2.5)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  (0.009397192462253184, tensor(90)) 

  #### Plot   ######################################################## 


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store"    ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter

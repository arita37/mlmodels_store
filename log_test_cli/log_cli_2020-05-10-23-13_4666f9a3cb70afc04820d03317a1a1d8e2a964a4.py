
  test_cli /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_cli', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_cli 

  # Testing Command Line System   





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4666f9a3cb70afc04820d03317a1a1d8e2a964a4', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '4666f9a3cb70afc04820d03317a1a1d8e2a964a4', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4666f9a3cb70afc04820d03317a1a1d8e2a964a4

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4666f9a3cb70afc04820d03317a1a1d8e2a964a4

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n']





 ************************************************************************************************************************
ml_models --do init  --path ztest/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
init

  Working Folder /home/runner/work/mlmodels/mlmodels 

  Config values {'model_trained': '/home/runner/work/mlmodels/mlmodels/model_trained/', 'dataset': '/home/runner/work/mlmodels/mlmodels/dataset/'} 

  Config path /home/runner/work/mlmodels/mlmodels/model_trained/ 





 ************************************************************************************************************************
ml_models --do model_list  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
model_list
/home/runner/work/mlmodels/mlmodels/mlmodels/
model_tf/temporal_fusion_google.py
model_tf/util.py
model_tf/__init__.py
model_tf/1_lstm.py
model_dev/__init__.py
model_sklearn/model_lightgbm.py
model_sklearn/__init__.py
model_sklearn/model_sklearn.py
model_tch/util_data.py
model_tch/pplm.py
model_tch/transformer_sentence.py
model_tch/__init__.py
model_tch/nbeats.py
model_tch/textcnn.py
model_tch/mlp.py
model_tch/pytorch_vae.py
model_tch/03_nbeats_dataloader.py
model_tch/torchhub.py
model_tch/util_transformer.py
model_tch/transformer_classifier.py
model_tch/matchzoo_models.py
model_gluon/util_autogluon.py
model_gluon/util.py
model_gluon/gluonts_model.py
model_gluon/__init__.py
model_gluon/fb_prophet.py
model_gluon/gluon_automl.py
model_flow/__init__.py
model_rank/__init__.py
model_keras/charcnn.py
model_keras/charcnn_zhang.py
model_keras/namentity_crm_bilstm_dataloader.py
model_keras/01_deepctr.py
model_keras/util.py
model_keras/__init__.py
model_keras/namentity_crm_bilstm.py
model_keras/nbeats.py
model_keras/textcnn.py
model_keras/keras_gan.py
model_keras/textcnn_dataloader.py
model_keras/02_cnn.py
model_keras/preprocess.py
model_keras/Autokeras.py
model_keras/armdn.py
model_keras/textvae.py





 ************************************************************************************************************************
ml_models  --do generate_config  --model_uri model_tf.1_lstm  --save_folder "ztest/"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
generate_config

  ztest/ 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
ztest/model_tf-1_lstm_config.json





 ************************************************************************************************************************
ml_models --do fit     --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
fit

  Fit 
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

  Save 
{'path': 'None', 'model_uri': 'model_tf/1_lstm.py'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 503, in main
    fit_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 411, in fit_cli
    save(module, model, sess, save_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 161, in save
    return module.save(model, session, save_pars, **kwarg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 192, in save
    save_tf(model, session, save_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 456, in save_tf
    os.makedirs(model_path, exist_ok=True)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/os.py", line 220, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
predict
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 507, in main
    predict_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 422, in predict_cli
    model, session = load(module, load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 154, in load
    return module.load(load_pars, **kwarg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 445, in load_tf
    saver = tf.compat.v1.train.Saver()  
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 865, in _build
    raise ValueError("No variables to save")
ValueError: No variables to save





 ************************************************************************************************************************
ml_models  --do test  --model_uri model_tf.1_lstm  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
test

  #### Module init   ############################################ 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  #### Model init   ############################################ 

  <mlmodels.model_tf.1_lstm.Model object at 0x7fcc36cc81d0> 

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
 [-0.11041464  0.12481052  0.0407635   0.16409574  0.17542174  0.03111502]
 [-0.03124873  0.12049093  0.0727565   0.01307441  0.11389384  0.13836589]
 [-0.07310653 -0.15280767 -0.0385585  -0.04470295  0.23166364 -0.11764826]
 [ 0.15690179 -0.03342742  0.04615156 -0.40585089  0.01216538  0.38622665]
 [-0.03720663  0.37501082 -0.01992735  0.33353901  0.18004455  0.24779333]
 [-0.17393631  0.24043372  0.16980498 -0.28731054  0.3953996  -0.16556361]
 [-0.2982733   0.24678347 -0.01988752 -0.14106332  0.0733116  -1.05806267]
 [ 0.46238282  0.1501976   0.30517271 -0.66094786  0.53486699 -0.56914413]
 [ 0.          0.          0.          0.          0.          0.        ]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 
model_tf.1_lstm
model_tf.1_lstm
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
{'loss': 0.634755402803421, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'
model_tf.1_lstm
model_tf.1_lstm
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
{'loss': 0.4754756949841976, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ml_models --do test  --model_uri "example/custom_model/1_lstm.py"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
test

  #### Module init   ############################################ 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  <module 'mlmodels.example.custom_model.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py'> 

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  #### Model init   ############################################ 

  <mlmodels.example.custom_model.1_lstm.Model object at 0x7fc9383c1278> 

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
 [-0.05913984  0.22858416  0.00775063 -0.02787498 -0.03430656  0.10148018]
 [ 0.06160572 -0.2626957  -0.1894145  -0.17556415 -0.08024645  0.28290939]
 [ 0.01264351 -0.10845775 -0.19489297 -0.1174463  -0.1776166   0.18146782]
 [-0.30529141 -0.10253249  0.06113402  0.07994999 -0.25448218  0.09374845]
 [ 0.30320913  0.08342127 -0.27656531  0.17119288 -0.08388664 -0.04274767]
 [ 0.2897287  -0.26235479  0.2742008  -0.02089711  0.03711822  0.14486344]
 [-0.72519857 -0.18340172  0.16579193 -0.63812453  0.49542353  0.11011156]
 [-0.27147472  0.71607065 -0.24217951  0.20269243  0.26311591  0.13414758]
 [ 0.          0.          0.          0.          0.          0.        ]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 
example/custom_model/1_lstm.py
example.custom_model.1_lstm.py
<module 'mlmodels.example.custom_model.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py'>
<module 'mlmodels.example.custom_model.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py'>

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
{'loss': 0.5038335397839546, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'
example/custom_model/1_lstm.py
example.custom_model.1_lstm.py
<module 'mlmodels.example.custom_model.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py'>
<module 'mlmodels.example.custom_model.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py'>

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
{'loss': 0.3665021751075983, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ml_optim --do search  --config_file template/optim_config.json  --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  ############# OPTIMIZATION Start  ############### 
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 388, in main
    optim_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 259, in optim_cli
    out_pars        = out_pars )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 54, in optim
    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
KeyError: 'engine_pars'





 ************************************************************************************************************************
ml_optim --do search  --config_file template/optim_config_prune.json   --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  ############# OPTIMIZATION Start  ############### 

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-10 23:13:55,325][0m Finished trial#0 resulted in value: 0.5972937345504761. Current best value is 0.5972937345504761 with parameters: {'learning_rate': 0.017840653660487423, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv
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

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-10 23:13:56,571][0m Finished trial#1 resulted in value: 0.5218324512243271. Current best value is 0.5218324512243271 with parameters: {'learning_rate': 0.016873301852584507, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv
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

 ################################### Optim, finished ###################################

  ### Save Stats   ########################################################## 

  ### Run Model with best   ################################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv
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

  #### Saving     ########################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/optim_1lstm/', 'model_type': 'model_tf', 'model_uri': 'model_tf-1_lstm'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 388, in main
    optim_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 259, in optim_cli
    out_pars        = out_pars )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 57, in optim
    out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 195, in optim_optuna
    module.save(model=model, session=sess, save_pars=save_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 192, in save
    save_tf(model, session, save_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 456, in save_tf
    os.makedirs(model_path, exist_ok=True)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/os.py", line 220, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ml_optim --do test   --model_uri model_tf.1_lstm   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-10 23:14:03,386][0m Finished trial#0 resulted in value: 2.547068804502487. Current best value is 2.547068804502487 with parameters: {'learning_rate': 0.05175819839016035, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv
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

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-10 23:14:04,690][0m Finished trial#1 resulted in value: 3.450262427330017. Current best value is 2.547068804502487 with parameters: {'learning_rate': 0.05175819839016035, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv
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

 ################################### Optim, finished ###################################

  ### Save Stats   ########################################################## 

  ### Run Model with best   ################################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv
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

  #### Saving     ########################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/optim_1lstm/', 'model_type': 'model_tf', 'model_uri': 'model_tf-1_lstm'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 382, in main
    test_json( path_json="template/optim_config_prune.json", config_mode= arg.config_mode )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 230, in test_json
    out_pars        = out_pars
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 57, in optim
    out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 195, in optim_optuna
    module.save(model=model, session=sess, save_pars=save_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 192, in save
    save_tf(model, session, save_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 456, in save_tf
    os.makedirs(model_path, exist_ok=True)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/os.py", line 220, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test02/model_list.json  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt

  dataset/json/benchmark.json 

  Custom benchmark 

  ['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'] 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_timeseries/test02/model_list.json 

  Model List [{'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}}] 

  


### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_fb/fb_prophet/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
INFO:numexpr.utils:NumExpr defaulting to 2 threads.
INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
Initial log joint probability = -192.039
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      99       9186.38     0.0272386        1207.2           1           1      123   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     199       10269.2     0.0242289       2566.31        0.89        0.89      233   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     299       10621.2     0.0237499       3262.95           1           1      343   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     399       10886.5     0.0339822       1343.14           1           1      459   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     499       11288.1    0.00255943       1266.79           1           1      580   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     599       11498.7     0.0166167       2146.51           1           1      698   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     699       11555.9     0.0104637       2039.91           1           1      812   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     799       11575.2    0.00955805       570.757           1           1      922   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     899       11630.7     0.0178715       1643.41      0.3435      0.3435     1036   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     999       11700.1      0.034504       2394.16           1           1     1146   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1099       11744.7   0.000237394       144.685           1           1     1258   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1199       11753.1    0.00188838       552.132      0.4814           1     1372   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1299         11758    0.00101299       262.652      0.7415      0.7415     1490   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1399         11761   0.000712302       157.258           1           1     1606   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1499       11781.3     0.0243264       931.457           1           1     1717   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1599       11791.1     0.0025484       550.483      0.7644      0.7644     1834   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1699       11797.7    0.00732868       810.153           1           1     1952   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1799       11802.5   0.000319611       98.1955     0.04871           1     2077   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1818       11803.2   5.97419e-05       246.505   3.588e-07       0.001     2142  LS failed, Hessian reset 
    1855       11803.6   0.000110613       144.447   1.529e-06       0.001     2225  LS failed, Hessian reset 
    1899       11804.3   0.000976631       305.295           1           1     2275   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1999       11805.4   4.67236e-05       72.2243      0.9487      0.9487     2391   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2033       11806.1   1.47341e-05       111.754   8.766e-08       0.001     2480  LS failed, Hessian reset 
    2099       11806.6   9.53816e-05       108.311      0.9684      0.9684     2563   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2151       11806.8   3.32394e-05       152.834   3.931e-07       0.001     2668  LS failed, Hessian reset 
    2199         11807    0.00273479       216.444           1           1     2723   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2299       11810.9    0.00793685       550.165           1           1     2837   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2399       11818.9     0.0134452       377.542           1           1     2952   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2499       11824.9     0.0041384       130.511           1           1     3060   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2525       11826.5   2.36518e-05       102.803   6.403e-08       0.001     3158  LS failed, Hessian reset 
    2599       11827.9   0.000370724       186.394      0.4637      0.4637     3242   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2606         11828   1.70497e-05       123.589     7.9e-08       0.001     3292  LS failed, Hessian reset 
    2699       11829.1    0.00168243       332.201           1           1     3407   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2709       11829.2   1.92694e-05       146.345   1.034e-07       0.001     3461  LS failed, Hessian reset 
    2746       11829.4   1.61976e-05       125.824   9.572e-08       0.001     3551  LS failed, Hessian reset 
    2799       11829.5    0.00491161       122.515           1           1     3615   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2899       11830.6   0.000250007       100.524           1           1     3742   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2999       11830.9    0.00236328       193.309           1           1     3889   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3099       11831.3   0.000309242       194.211      0.7059      0.7059     4015   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3199       11831.4    1.3396e-05       91.8042      0.9217      0.9217     4136   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3299       11831.6   0.000373334       77.3538      0.3184           1     4256   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3399       11831.8   0.000125272       64.7127           1           1     4379   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3499         11832     0.0010491       69.8273           1           1     4503   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3553       11832.1   1.09422e-05       89.3197   8.979e-08       0.001     4612  LS failed, Hessian reset 
    3584       11832.1   8.65844e-07       55.9367      0.4252      0.4252     4658   
Optimization terminated normally: 
  Convergence detected: relative gradient magnitude is below tolerance
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fccf8d10198> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:14:21.944019
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 23:14:21.947835
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 23:14:21.950997
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 23:14:21.954129
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -18.2877
metric_name                                             r2_score
Name: 3, dtype: object 

  


### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/armdn/'} 

  #### Setup Model   ############################################## 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_probability/python/distributions/mixture.py:154: Categorical.event_size (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-05-19.
Instructions for updating:
The `event_size` property is deprecated.  Use `num_categories` instead.  They have the same value, but `event_size` is misnamed.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
LSTM_1 (LSTM)                (None, 60, 300)           362400    
_________________________________________________________________
LSTM_2 (LSTM)                (None, 60, 200)           400800    
_________________________________________________________________
LSTM_3 (LSTM)                (None, 60, 24)            21600     
_________________________________________________________________
LSTM_4 (LSTM)                (None, 12)                1776      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                130       
_________________________________________________________________
mdn_1 (MDN)                  (None, 363)               3993      
=================================================================
Total params: 790,699
Trainable params: 790,699
Non-trainable params: 0
_________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fccf8d101d0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353917.0625
Epoch 2/10

1/1 [==============================] - 0s 98ms/step - loss: 241260.8906
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 130384.9453
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 57599.9180
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 26606.6621
Epoch 6/10

1/1 [==============================] - 0s 105ms/step - loss: 14066.0928
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 8452.1172
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 5689.5410
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 4161.9990
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 3271.5439

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.18647408e+00 -1.67590272e+00 -9.85079050e-01  4.28107738e-01
   5.32304764e-01 -4.34371263e-01 -9.36429739e-01  3.57743919e-01
   1.61896646e-02 -5.57481766e-01 -1.94415629e-01 -3.99543703e-01
  -5.97643375e-01 -8.76274943e-01 -1.06256235e+00 -2.02273309e-01
   2.56223869e+00  1.05140114e+00  4.31937814e-01 -1.75147605e+00
  -3.37932885e-01 -6.79636419e-01 -1.87851250e-01  2.01435328e-01
   6.78577185e-01  2.23284674e+00  6.95898414e-01 -6.15432143e-01
   7.11742818e-01 -5.30305803e-01 -7.27553368e-02 -2.02349305e+00
  -2.02759624e+00  3.02434266e-01 -2.90479064e-01 -2.89100337e+00
   2.34991550e+00 -2.56468654e-02 -6.16189480e-01  1.69312716e+00
   1.48175204e+00  1.37815237e+00  6.85407281e-01  1.61773002e+00
  -1.33769155e-01  1.25392163e+00 -3.18936110e-01 -4.96732056e-01
   1.03434575e+00 -7.71221042e-01 -6.68127298e-01 -1.94740891e-02
   1.32852733e+00  7.08281159e-01 -2.83210182e+00 -4.16800916e-01
   7.60712743e-01 -2.62964439e+00  2.07679439e+00  4.41708267e-01
  -2.21053779e-01  1.19923086e+01  1.11666393e+01  1.25586195e+01
   1.19297934e+01  1.42303486e+01  1.48137054e+01  1.28401690e+01
   1.02031927e+01  1.12961731e+01  1.38671741e+01  1.21274347e+01
   1.40124083e+01  1.24338713e+01  1.09277325e+01  1.21810780e+01
   1.34532375e+01  1.12455568e+01  1.18295975e+01  1.14867678e+01
   1.40814276e+01  1.34817924e+01  1.27373838e+01  1.26165056e+01
   1.37368784e+01  1.22169476e+01  1.04364357e+01  1.34648314e+01
   1.17148304e+01  1.27639151e+01  1.17182188e+01  1.23259239e+01
   1.26125431e+01  1.15117216e+01  1.11676149e+01  1.28483686e+01
   1.04930248e+01  1.25770454e+01  1.36228724e+01  1.34405003e+01
   1.23592434e+01  1.15042868e+01  1.32044611e+01  1.41427517e+01
   1.11827059e+01  1.21036148e+01  1.36394119e+01  1.23812437e+01
   1.26185112e+01  1.06030579e+01  1.25527296e+01  1.17684698e+01
   1.47757015e+01  1.44378624e+01  1.47826433e+01  1.37054996e+01
   1.17780476e+01  1.15438967e+01  1.23058033e+01  1.09200039e+01
   1.35964775e+00 -7.76356459e-03  9.36474323e-01  7.40417004e-01
  -2.76256919e-01 -2.88845032e-01 -7.76897132e-01  1.59163272e+00
  -9.95889187e-01 -4.32065219e-01 -5.57212353e-01  5.44546604e-01
  -2.25522041e-01 -4.09287363e-01  3.07621360e-02  2.69089580e+00
  -1.21741927e+00  7.20857620e-01 -2.76298165e+00 -2.63684940e+00
   3.07039797e-01  5.18142879e-01 -9.30696845e-01  2.73166597e-01
   2.13052392e+00 -1.25741506e+00 -1.10185015e+00 -2.18889475e+00
   7.11739540e-01  4.12963629e-01  2.13723838e-01  8.54362607e-01
  -3.86198938e-01 -1.23991740e+00 -1.81439626e+00 -3.15077960e-01
   1.65911591e+00 -1.57351685e+00  1.34256387e+00 -4.72063333e-01
  -5.49597383e-01 -1.09760022e+00  3.38304371e-01  2.69205689e-01
   4.53608453e-01 -1.28736091e+00  1.41731608e+00 -5.97111821e-01
   5.06856680e-01  1.35397148e+00 -1.49892259e+00  1.05034447e+00
  -1.04300547e+00 -6.52067304e-01  7.94759989e-01 -6.06906414e-02
  -8.28363180e-01 -1.73677671e+00  1.13695073e+00  8.41643691e-01
   2.54467726e-01  2.47141898e-01  8.70671272e-02  3.19329691e+00
   1.27580953e+00  1.50534809e-01  4.50397491e-01  1.11137629e-01
   1.18707204e+00  1.22690082e-01  1.27189994e-01  3.09523058e+00
   3.90215778e+00  2.00913382e+00  7.97130108e-01  2.54768372e-01
   5.29471993e-01  5.00305831e-01  1.99983823e+00  7.03075469e-01
   6.63653910e-01  1.79412425e-01  2.49563313e+00  4.70205188e-01
   7.20507681e-01  8.64114761e-02  2.67072201e+00  6.63438857e-01
   1.83387327e+00  2.88504362e+00  4.63005900e-02  1.61534786e-01
   1.13133371e-01  1.29726768e-01  1.91304862e-01  2.34278297e+00
   2.06852019e-01  2.07514572e+00  7.97106326e-01  4.88869846e-01
   2.12390661e+00  4.70474899e-01  2.61275411e-01  1.50059187e+00
   2.02765465e+00  3.42361736e+00  3.53883076e+00  3.45517457e-01
   2.69021010e+00  4.59461391e-01  5.94793081e-01  2.29300976e+00
   2.46169806e-01  1.12928116e+00  7.81494975e-02  3.10208738e-01
   1.20907974e+00  2.63784289e-01  8.10482144e-01  8.03618252e-01
   7.68866777e-01  1.49312162e+01  1.29089699e+01  1.37468262e+01
   1.45579157e+01  1.06657591e+01  9.69433880e+00  1.34155045e+01
   1.37649574e+01  1.12120924e+01  1.36753683e+01  1.44626770e+01
   1.40655413e+01  1.31086111e+01  1.26475630e+01  1.42487097e+01
   1.43273096e+01  1.17338028e+01  1.24246836e+01  1.14723215e+01
   1.40052509e+01  1.10772009e+01  1.21673889e+01  1.19989605e+01
   1.19115019e+01  1.29896221e+01  1.17868004e+01  1.23839607e+01
   1.22603149e+01  1.05757132e+01  1.20829096e+01  1.33144426e+01
   1.12294235e+01  1.40035276e+01  1.21239672e+01  1.02954235e+01
   1.16111002e+01  1.11083574e+01  1.35714588e+01  1.13836565e+01
   1.40168657e+01  1.15951052e+01  1.36455107e+01  1.34780140e+01
   1.11537018e+01  1.26997614e+01  1.31406212e+01  1.21060209e+01
   1.23454618e+01  1.29131126e+01  1.13600416e+01  1.15377092e+01
   1.30083752e+01  1.22905273e+01  1.24046974e+01  1.22133446e+01
   1.33302755e+01  1.25637379e+01  1.28772745e+01  1.30895367e+01
   7.47965276e-01  2.05952978e+00  2.59223938e+00  7.75077939e-01
   7.10415483e-01  2.34581053e-01  3.67184937e-01  1.96149087e+00
   2.11017191e-01  1.26302052e+00  2.41402054e+00  2.88079453e+00
   7.37971663e-01  1.05884409e+00  5.10680497e-01  4.62465048e-01
   1.53133106e+00  1.60224974e-01  1.44336200e+00  3.14793348e+00
   4.21337843e-01  6.13408267e-01  1.31758845e+00  1.36262119e-01
   2.41764545e+00  5.06344914e-01  3.85483921e-01  2.51421571e-01
   1.57915390e+00  4.82245862e-01  8.12981069e-01  1.50195384e+00
   5.94166338e-01  1.57918870e+00  2.43174732e-01  1.76758087e+00
   8.89293611e-01  1.87304723e+00  3.34411860e-01  8.19040060e-01
   3.16189885e-01  1.87527347e+00  1.00184917e+00  2.20605755e+00
   8.53256166e-01  4.17402744e-01  3.07876587e+00  9.41703856e-01
   7.59875178e-01  4.09508109e-01  1.91386747e+00  1.44731736e+00
   6.82003498e-01  7.01587141e-01  1.79988492e+00  9.34714258e-01
   1.88496172e-01  3.49054718e+00  7.77562678e-01  9.48290706e-01
  -1.24111309e+01  1.80433922e+01 -1.21994829e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:14:31.240117
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   90.4182
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 23:14:31.243884
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    8202.6
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 23:14:31.251888
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   90.0451
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 23:14:31.255186
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -733.601
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140517987298216
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140516759296488
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140516759296992
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140516759297496
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140516759298000
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140516759323144

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fccf2990fd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.463520
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.431181
grad_step = 000002, loss = 0.414609
grad_step = 000003, loss = 0.398749
grad_step = 000004, loss = 0.381928
grad_step = 000005, loss = 0.364060
grad_step = 000006, loss = 0.347061
grad_step = 000007, loss = 0.335470
grad_step = 000008, loss = 0.330065
grad_step = 000009, loss = 0.320801
grad_step = 000010, loss = 0.307567
grad_step = 000011, loss = 0.296734
grad_step = 000012, loss = 0.288804
grad_step = 000013, loss = 0.281263
grad_step = 000014, loss = 0.273185
grad_step = 000015, loss = 0.264272
grad_step = 000016, loss = 0.254994
grad_step = 000017, loss = 0.245991
grad_step = 000018, loss = 0.237726
grad_step = 000019, loss = 0.230174
grad_step = 000020, loss = 0.222691
grad_step = 000021, loss = 0.214695
grad_step = 000022, loss = 0.206421
grad_step = 000023, loss = 0.198512
grad_step = 000024, loss = 0.191229
grad_step = 000025, loss = 0.184322
grad_step = 000026, loss = 0.177399
grad_step = 000027, loss = 0.170298
grad_step = 000028, loss = 0.163073
grad_step = 000029, loss = 0.155978
grad_step = 000030, loss = 0.149311
grad_step = 000031, loss = 0.142989
grad_step = 000032, loss = 0.136723
grad_step = 000033, loss = 0.128968
grad_step = 000034, loss = 0.121086
grad_step = 000035, loss = 0.114176
grad_step = 000036, loss = 0.108637
grad_step = 000037, loss = 0.102835
grad_step = 000038, loss = 0.097146
grad_step = 000039, loss = 0.091355
grad_step = 000040, loss = 0.085496
grad_step = 000041, loss = 0.080526
grad_step = 000042, loss = 0.075863
grad_step = 000043, loss = 0.070895
grad_step = 000044, loss = 0.065967
grad_step = 000045, loss = 0.061382
grad_step = 000046, loss = 0.057279
grad_step = 000047, loss = 0.053289
grad_step = 000048, loss = 0.049364
grad_step = 000049, loss = 0.045586
grad_step = 000050, loss = 0.041086
grad_step = 000051, loss = 0.036868
grad_step = 000052, loss = 0.033230
grad_step = 000053, loss = 0.030434
grad_step = 000054, loss = 0.027975
grad_step = 000055, loss = 0.025511
grad_step = 000056, loss = 0.022956
grad_step = 000057, loss = 0.020513
grad_step = 000058, loss = 0.018388
grad_step = 000059, loss = 0.016329
grad_step = 000060, loss = 0.014513
grad_step = 000061, loss = 0.012949
grad_step = 000062, loss = 0.011558
grad_step = 000063, loss = 0.010221
grad_step = 000064, loss = 0.009036
grad_step = 000065, loss = 0.007977
grad_step = 000066, loss = 0.007116
grad_step = 000067, loss = 0.006338
grad_step = 000068, loss = 0.005614
grad_step = 000069, loss = 0.004966
grad_step = 000070, loss = 0.004490
grad_step = 000071, loss = 0.004102
grad_step = 000072, loss = 0.003774
grad_step = 000073, loss = 0.003475
grad_step = 000074, loss = 0.003235
grad_step = 000075, loss = 0.003038
grad_step = 000076, loss = 0.002897
grad_step = 000077, loss = 0.002806
grad_step = 000078, loss = 0.002734
grad_step = 000079, loss = 0.002684
grad_step = 000080, loss = 0.002639
grad_step = 000081, loss = 0.002613
grad_step = 000082, loss = 0.002597
grad_step = 000083, loss = 0.002583
grad_step = 000084, loss = 0.002565
grad_step = 000085, loss = 0.002549
grad_step = 000086, loss = 0.002533
grad_step = 000087, loss = 0.002516
grad_step = 000088, loss = 0.002511
grad_step = 000089, loss = 0.002509
grad_step = 000090, loss = 0.002497
grad_step = 000091, loss = 0.002498
grad_step = 000092, loss = 0.002538
grad_step = 000093, loss = 0.002570
grad_step = 000094, loss = 0.002535
grad_step = 000095, loss = 0.002388
grad_step = 000096, loss = 0.002298
grad_step = 000097, loss = 0.002335
grad_step = 000098, loss = 0.002360
grad_step = 000099, loss = 0.002284
grad_step = 000100, loss = 0.002192
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002208
grad_step = 000102, loss = 0.002238
grad_step = 000103, loss = 0.002179
grad_step = 000104, loss = 0.002114
grad_step = 000105, loss = 0.002124
grad_step = 000106, loss = 0.002145
grad_step = 000107, loss = 0.002106
grad_step = 000108, loss = 0.002062
grad_step = 000109, loss = 0.002066
grad_step = 000110, loss = 0.002084
grad_step = 000111, loss = 0.002061
grad_step = 000112, loss = 0.002029
grad_step = 000113, loss = 0.002027
grad_step = 000114, loss = 0.002037
grad_step = 000115, loss = 0.002031
grad_step = 000116, loss = 0.002008
grad_step = 000117, loss = 0.001996
grad_step = 000118, loss = 0.002001
grad_step = 000119, loss = 0.002004
grad_step = 000120, loss = 0.001997
grad_step = 000121, loss = 0.001981
grad_step = 000122, loss = 0.001970
grad_step = 000123, loss = 0.001968
grad_step = 000124, loss = 0.001971
grad_step = 000125, loss = 0.001971
grad_step = 000126, loss = 0.001966
grad_step = 000127, loss = 0.001957
grad_step = 000128, loss = 0.001946
grad_step = 000129, loss = 0.001938
grad_step = 000130, loss = 0.001932
grad_step = 000131, loss = 0.001929
grad_step = 000132, loss = 0.001927
grad_step = 000133, loss = 0.001927
grad_step = 000134, loss = 0.001930
grad_step = 000135, loss = 0.001939
grad_step = 000136, loss = 0.001963
grad_step = 000137, loss = 0.002002
grad_step = 000138, loss = 0.002069
grad_step = 000139, loss = 0.002086
grad_step = 000140, loss = 0.002048
grad_step = 000141, loss = 0.001939
grad_step = 000142, loss = 0.001889
grad_step = 000143, loss = 0.001932
grad_step = 000144, loss = 0.001983
grad_step = 000145, loss = 0.001959
grad_step = 000146, loss = 0.001890
grad_step = 000147, loss = 0.001869
grad_step = 000148, loss = 0.001908
grad_step = 000149, loss = 0.001939
grad_step = 000150, loss = 0.001922
grad_step = 000151, loss = 0.001875
grad_step = 000152, loss = 0.001852
grad_step = 000153, loss = 0.001865
grad_step = 000154, loss = 0.001886
grad_step = 000155, loss = 0.001885
grad_step = 000156, loss = 0.001853
grad_step = 000157, loss = 0.001832
grad_step = 000158, loss = 0.001839
grad_step = 000159, loss = 0.001853
grad_step = 000160, loss = 0.001851
grad_step = 000161, loss = 0.001831
grad_step = 000162, loss = 0.001818
grad_step = 000163, loss = 0.001820
grad_step = 000164, loss = 0.001826
grad_step = 000165, loss = 0.001826
grad_step = 000166, loss = 0.001816
grad_step = 000167, loss = 0.001805
grad_step = 000168, loss = 0.001803
grad_step = 000169, loss = 0.001807
grad_step = 000170, loss = 0.001812
grad_step = 000171, loss = 0.001810
grad_step = 000172, loss = 0.001807
grad_step = 000173, loss = 0.001803
grad_step = 000174, loss = 0.001807
grad_step = 000175, loss = 0.001816
grad_step = 000176, loss = 0.001830
grad_step = 000177, loss = 0.001831
grad_step = 000178, loss = 0.001827
grad_step = 000179, loss = 0.001808
grad_step = 000180, loss = 0.001802
grad_step = 000181, loss = 0.001805
grad_step = 000182, loss = 0.001815
grad_step = 000183, loss = 0.001804
grad_step = 000184, loss = 0.001783
grad_step = 000185, loss = 0.001759
grad_step = 000186, loss = 0.001753
grad_step = 000187, loss = 0.001760
grad_step = 000188, loss = 0.001767
grad_step = 000189, loss = 0.001762
grad_step = 000190, loss = 0.001748
grad_step = 000191, loss = 0.001737
grad_step = 000192, loss = 0.001735
grad_step = 000193, loss = 0.001740
grad_step = 000194, loss = 0.001745
grad_step = 000195, loss = 0.001746
grad_step = 000196, loss = 0.001744
grad_step = 000197, loss = 0.001744
grad_step = 000198, loss = 0.001756
grad_step = 000199, loss = 0.001809
grad_step = 000200, loss = 0.001915
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002162
grad_step = 000202, loss = 0.002195
grad_step = 000203, loss = 0.002086
grad_step = 000204, loss = 0.001748
grad_step = 000205, loss = 0.001828
grad_step = 000206, loss = 0.002019
grad_step = 000207, loss = 0.001801
grad_step = 000208, loss = 0.001761
grad_step = 000209, loss = 0.001906
grad_step = 000210, loss = 0.001768
grad_step = 000211, loss = 0.001737
grad_step = 000212, loss = 0.001838
grad_step = 000213, loss = 0.001732
grad_step = 000214, loss = 0.001716
grad_step = 000215, loss = 0.001797
grad_step = 000216, loss = 0.001712
grad_step = 000217, loss = 0.001699
grad_step = 000218, loss = 0.001760
grad_step = 000219, loss = 0.001698
grad_step = 000220, loss = 0.001687
grad_step = 000221, loss = 0.001723
grad_step = 000222, loss = 0.001681
grad_step = 000223, loss = 0.001678
grad_step = 000224, loss = 0.001700
grad_step = 000225, loss = 0.001668
grad_step = 000226, loss = 0.001663
grad_step = 000227, loss = 0.001682
grad_step = 000228, loss = 0.001660
grad_step = 000229, loss = 0.001647
grad_step = 000230, loss = 0.001661
grad_step = 000231, loss = 0.001653
grad_step = 000232, loss = 0.001636
grad_step = 000233, loss = 0.001641
grad_step = 000234, loss = 0.001642
grad_step = 000235, loss = 0.001629
grad_step = 000236, loss = 0.001624
grad_step = 000237, loss = 0.001629
grad_step = 000238, loss = 0.001624
grad_step = 000239, loss = 0.001613
grad_step = 000240, loss = 0.001612
grad_step = 000241, loss = 0.001614
grad_step = 000242, loss = 0.001606
grad_step = 000243, loss = 0.001599
grad_step = 000244, loss = 0.001598
grad_step = 000245, loss = 0.001598
grad_step = 000246, loss = 0.001592
grad_step = 000247, loss = 0.001586
grad_step = 000248, loss = 0.001586
grad_step = 000249, loss = 0.001591
grad_step = 000250, loss = 0.001605
grad_step = 000251, loss = 0.001638
grad_step = 000252, loss = 0.001684
grad_step = 000253, loss = 0.001707
grad_step = 000254, loss = 0.001686
grad_step = 000255, loss = 0.001613
grad_step = 000256, loss = 0.001564
grad_step = 000257, loss = 0.001583
grad_step = 000258, loss = 0.001623
grad_step = 000259, loss = 0.001629
grad_step = 000260, loss = 0.001596
grad_step = 000261, loss = 0.001556
grad_step = 000262, loss = 0.001544
grad_step = 000263, loss = 0.001564
grad_step = 000264, loss = 0.001586
grad_step = 000265, loss = 0.001589
grad_step = 000266, loss = 0.001574
grad_step = 000267, loss = 0.001555
grad_step = 000268, loss = 0.001542
grad_step = 000269, loss = 0.001537
grad_step = 000270, loss = 0.001531
grad_step = 000271, loss = 0.001529
grad_step = 000272, loss = 0.001532
grad_step = 000273, loss = 0.001536
grad_step = 000274, loss = 0.001540
grad_step = 000275, loss = 0.001531
grad_step = 000276, loss = 0.001518
grad_step = 000277, loss = 0.001506
grad_step = 000278, loss = 0.001499
grad_step = 000279, loss = 0.001499
grad_step = 000280, loss = 0.001500
grad_step = 000281, loss = 0.001499
grad_step = 000282, loss = 0.001496
grad_step = 000283, loss = 0.001496
grad_step = 000284, loss = 0.001497
grad_step = 000285, loss = 0.001501
grad_step = 000286, loss = 0.001507
grad_step = 000287, loss = 0.001512
grad_step = 000288, loss = 0.001516
grad_step = 000289, loss = 0.001523
grad_step = 000290, loss = 0.001525
grad_step = 000291, loss = 0.001531
grad_step = 000292, loss = 0.001518
grad_step = 000293, loss = 0.001501
grad_step = 000294, loss = 0.001475
grad_step = 000295, loss = 0.001457
grad_step = 000296, loss = 0.001452
grad_step = 000297, loss = 0.001459
grad_step = 000298, loss = 0.001470
grad_step = 000299, loss = 0.001474
grad_step = 000300, loss = 0.001476
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001469
grad_step = 000302, loss = 0.001468
grad_step = 000303, loss = 0.001475
grad_step = 000304, loss = 0.001487
grad_step = 000305, loss = 0.001495
grad_step = 000306, loss = 0.001495
grad_step = 000307, loss = 0.001487
grad_step = 000308, loss = 0.001470
grad_step = 000309, loss = 0.001451
grad_step = 000310, loss = 0.001439
grad_step = 000311, loss = 0.001431
grad_step = 000312, loss = 0.001429
grad_step = 000313, loss = 0.001427
grad_step = 000314, loss = 0.001423
grad_step = 000315, loss = 0.001418
grad_step = 000316, loss = 0.001413
grad_step = 000317, loss = 0.001410
grad_step = 000318, loss = 0.001410
grad_step = 000319, loss = 0.001416
grad_step = 000320, loss = 0.001423
grad_step = 000321, loss = 0.001438
grad_step = 000322, loss = 0.001448
grad_step = 000323, loss = 0.001466
grad_step = 000324, loss = 0.001462
grad_step = 000325, loss = 0.001454
grad_step = 000326, loss = 0.001426
grad_step = 000327, loss = 0.001403
grad_step = 000328, loss = 0.001393
grad_step = 000329, loss = 0.001396
grad_step = 000330, loss = 0.001407
grad_step = 000331, loss = 0.001410
grad_step = 000332, loss = 0.001407
grad_step = 000333, loss = 0.001392
grad_step = 000334, loss = 0.001377
grad_step = 000335, loss = 0.001367
grad_step = 000336, loss = 0.001364
grad_step = 000337, loss = 0.001367
grad_step = 000338, loss = 0.001373
grad_step = 000339, loss = 0.001380
grad_step = 000340, loss = 0.001382
grad_step = 000341, loss = 0.001385
grad_step = 000342, loss = 0.001381
grad_step = 000343, loss = 0.001379
grad_step = 000344, loss = 0.001372
grad_step = 000345, loss = 0.001369
grad_step = 000346, loss = 0.001365
grad_step = 000347, loss = 0.001367
grad_step = 000348, loss = 0.001373
grad_step = 000349, loss = 0.001388
grad_step = 000350, loss = 0.001410
grad_step = 000351, loss = 0.001437
grad_step = 000352, loss = 0.001464
grad_step = 000353, loss = 0.001471
grad_step = 000354, loss = 0.001455
grad_step = 000355, loss = 0.001405
grad_step = 000356, loss = 0.001352
grad_step = 000357, loss = 0.001321
grad_step = 000358, loss = 0.001324
grad_step = 000359, loss = 0.001348
grad_step = 000360, loss = 0.001369
grad_step = 000361, loss = 0.001373
grad_step = 000362, loss = 0.001354
grad_step = 000363, loss = 0.001330
grad_step = 000364, loss = 0.001317
grad_step = 000365, loss = 0.001319
grad_step = 000366, loss = 0.001332
grad_step = 000367, loss = 0.001338
grad_step = 000368, loss = 0.001337
grad_step = 000369, loss = 0.001322
grad_step = 000370, loss = 0.001308
grad_step = 000371, loss = 0.001301
grad_step = 000372, loss = 0.001303
grad_step = 000373, loss = 0.001308
grad_step = 000374, loss = 0.001311
grad_step = 000375, loss = 0.001307
grad_step = 000376, loss = 0.001300
grad_step = 000377, loss = 0.001292
grad_step = 000378, loss = 0.001290
grad_step = 000379, loss = 0.001291
grad_step = 000380, loss = 0.001297
grad_step = 000381, loss = 0.001300
grad_step = 000382, loss = 0.001304
grad_step = 000383, loss = 0.001303
grad_step = 000384, loss = 0.001308
grad_step = 000385, loss = 0.001308
grad_step = 000386, loss = 0.001316
grad_step = 000387, loss = 0.001313
grad_step = 000388, loss = 0.001313
grad_step = 000389, loss = 0.001297
grad_step = 000390, loss = 0.001279
grad_step = 000391, loss = 0.001259
grad_step = 000392, loss = 0.001246
grad_step = 000393, loss = 0.001242
grad_step = 000394, loss = 0.001246
grad_step = 000395, loss = 0.001254
grad_step = 000396, loss = 0.001261
grad_step = 000397, loss = 0.001271
grad_step = 000398, loss = 0.001274
grad_step = 000399, loss = 0.001279
grad_step = 000400, loss = 0.001273
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001268
grad_step = 000402, loss = 0.001253
grad_step = 000403, loss = 0.001241
grad_step = 000404, loss = 0.001230
grad_step = 000405, loss = 0.001224
grad_step = 000406, loss = 0.001222
grad_step = 000407, loss = 0.001223
grad_step = 000408, loss = 0.001224
grad_step = 000409, loss = 0.001224
grad_step = 000410, loss = 0.001225
grad_step = 000411, loss = 0.001225
grad_step = 000412, loss = 0.001229
grad_step = 000413, loss = 0.001231
grad_step = 000414, loss = 0.001239
grad_step = 000415, loss = 0.001243
grad_step = 000416, loss = 0.001253
grad_step = 000417, loss = 0.001253
grad_step = 000418, loss = 0.001257
grad_step = 000419, loss = 0.001247
grad_step = 000420, loss = 0.001241
grad_step = 000421, loss = 0.001225
grad_step = 000422, loss = 0.001215
grad_step = 000423, loss = 0.001203
grad_step = 000424, loss = 0.001197
grad_step = 000425, loss = 0.001194
grad_step = 000426, loss = 0.001195
grad_step = 000427, loss = 0.001205
grad_step = 000428, loss = 0.001213
grad_step = 000429, loss = 0.001229
grad_step = 000430, loss = 0.001228
grad_step = 000431, loss = 0.001224
grad_step = 000432, loss = 0.001200
grad_step = 000433, loss = 0.001177
grad_step = 000434, loss = 0.001157
grad_step = 000435, loss = 0.001148
grad_step = 000436, loss = 0.001152
grad_step = 000437, loss = 0.001161
grad_step = 000438, loss = 0.001171
grad_step = 000439, loss = 0.001175
grad_step = 000440, loss = 0.001178
grad_step = 000441, loss = 0.001171
grad_step = 000442, loss = 0.001166
grad_step = 000443, loss = 0.001154
grad_step = 000444, loss = 0.001144
grad_step = 000445, loss = 0.001134
grad_step = 000446, loss = 0.001128
grad_step = 000447, loss = 0.001126
grad_step = 000448, loss = 0.001128
grad_step = 000449, loss = 0.001135
grad_step = 000450, loss = 0.001142
grad_step = 000451, loss = 0.001154
grad_step = 000452, loss = 0.001165
grad_step = 000453, loss = 0.001187
grad_step = 000454, loss = 0.001200
grad_step = 000455, loss = 0.001225
grad_step = 000456, loss = 0.001222
grad_step = 000457, loss = 0.001222
grad_step = 000458, loss = 0.001184
grad_step = 000459, loss = 0.001145
grad_step = 000460, loss = 0.001105
grad_step = 000461, loss = 0.001087
grad_step = 000462, loss = 0.001091
grad_step = 000463, loss = 0.001108
grad_step = 000464, loss = 0.001126
grad_step = 000465, loss = 0.001125
grad_step = 000466, loss = 0.001116
grad_step = 000467, loss = 0.001095
grad_step = 000468, loss = 0.001083
grad_step = 000469, loss = 0.001083
grad_step = 000470, loss = 0.001097
grad_step = 000471, loss = 0.001117
grad_step = 000472, loss = 0.001138
grad_step = 000473, loss = 0.001153
grad_step = 000474, loss = 0.001172
grad_step = 000475, loss = 0.001182
grad_step = 000476, loss = 0.001226
grad_step = 000477, loss = 0.001215
grad_step = 000478, loss = 0.001204
grad_step = 000479, loss = 0.001119
grad_step = 000480, loss = 0.001059
grad_step = 000481, loss = 0.001052
grad_step = 000482, loss = 0.001089
grad_step = 000483, loss = 0.001122
grad_step = 000484, loss = 0.001097
grad_step = 000485, loss = 0.001059
grad_step = 000486, loss = 0.001028
grad_step = 000487, loss = 0.001033
grad_step = 000488, loss = 0.001058
grad_step = 000489, loss = 0.001066
grad_step = 000490, loss = 0.001058
grad_step = 000491, loss = 0.001029
grad_step = 000492, loss = 0.001009
grad_step = 000493, loss = 0.001007
grad_step = 000494, loss = 0.001017
grad_step = 000495, loss = 0.001026
grad_step = 000496, loss = 0.001021
grad_step = 000497, loss = 0.001009
grad_step = 000498, loss = 0.000993
grad_step = 000499, loss = 0.000984
grad_step = 000500, loss = 0.000985
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000990
Finished.

  #### Inference Need return ypred, ytrue ######################### 
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:14:50.255664
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.276538
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 23:14:50.261482
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.223284
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 23:14:50.268730
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.126968
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 23:14:50.274180
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -2.39288
metric_name                                             r2_score
Name: 11, dtype: object 

  


### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/ 

                        date_run  ...            metric_name
0   2020-05-10 23:14:21.944019  ...    mean_absolute_error
1   2020-05-10 23:14:21.947835  ...     mean_squared_error
2   2020-05-10 23:14:21.950997  ...  median_absolute_error
3   2020-05-10 23:14:21.954129  ...               r2_score
4   2020-05-10 23:14:31.240117  ...    mean_absolute_error
5   2020-05-10 23:14:31.243884  ...     mean_squared_error
6   2020-05-10 23:14:31.251888  ...  median_absolute_error
7   2020-05-10 23:14:31.255186  ...               r2_score
8   2020-05-10 23:14:50.255664  ...    mean_absolute_error
9   2020-05-10 23:14:50.261482  ...     mean_squared_error
10  2020-05-10 23:14:50.268730  ...  median_absolute_error
11  2020-05-10 23:14:50.274180  ...               r2_score

[12 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range





 ************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt

  dataset/json/benchmark.json 

  Custom benchmark 

  ['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'] 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_timeseries/test01/ 

  Model List [{'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}] 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140282951782752
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140281672497528
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140281672498032
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140281672498536
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140281672499040
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140281672634776

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f962b82cfd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.458991
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.430553
grad_step = 000002, loss = 0.408496
grad_step = 000003, loss = 0.383494
grad_step = 000004, loss = 0.353833
grad_step = 000005, loss = 0.322664
grad_step = 000006, loss = 0.300885
grad_step = 000007, loss = 0.292579
grad_step = 000008, loss = 0.279718
grad_step = 000009, loss = 0.260811
grad_step = 000010, loss = 0.242617
grad_step = 000011, loss = 0.229666
grad_step = 000012, loss = 0.222949
grad_step = 000013, loss = 0.215288
grad_step = 000014, loss = 0.204764
grad_step = 000015, loss = 0.193531
grad_step = 000016, loss = 0.183595
grad_step = 000017, loss = 0.175583
grad_step = 000018, loss = 0.168272
grad_step = 000019, loss = 0.160229
grad_step = 000020, loss = 0.151588
grad_step = 000021, loss = 0.143296
grad_step = 000022, loss = 0.135577
grad_step = 000023, loss = 0.127885
grad_step = 000024, loss = 0.120046
grad_step = 000025, loss = 0.112509
grad_step = 000026, loss = 0.105530
grad_step = 000027, loss = 0.098986
grad_step = 000028, loss = 0.092833
grad_step = 000029, loss = 0.086934
grad_step = 000030, loss = 0.081127
grad_step = 000031, loss = 0.075575
grad_step = 000032, loss = 0.070255
grad_step = 000033, loss = 0.064959
grad_step = 000034, loss = 0.059944
grad_step = 000035, loss = 0.055451
grad_step = 000036, loss = 0.051228
grad_step = 000037, loss = 0.047111
grad_step = 000038, loss = 0.043279
grad_step = 000039, loss = 0.039841
grad_step = 000040, loss = 0.036656
grad_step = 000041, loss = 0.033644
grad_step = 000042, loss = 0.030783
grad_step = 000043, loss = 0.028038
grad_step = 000044, loss = 0.025508
grad_step = 000045, loss = 0.023305
grad_step = 000046, loss = 0.021347
grad_step = 000047, loss = 0.019480
grad_step = 000048, loss = 0.017700
grad_step = 000049, loss = 0.016118
grad_step = 000050, loss = 0.014716
grad_step = 000051, loss = 0.013395
grad_step = 000052, loss = 0.012195
grad_step = 000053, loss = 0.011134
grad_step = 000054, loss = 0.010151
grad_step = 000055, loss = 0.009257
grad_step = 000056, loss = 0.008453
grad_step = 000057, loss = 0.007720
grad_step = 000058, loss = 0.007085
grad_step = 000059, loss = 0.006525
grad_step = 000060, loss = 0.005998
grad_step = 000061, loss = 0.005520
grad_step = 000062, loss = 0.005108
grad_step = 000063, loss = 0.004742
grad_step = 000064, loss = 0.004416
grad_step = 000065, loss = 0.004132
grad_step = 000066, loss = 0.003880
grad_step = 000067, loss = 0.003664
grad_step = 000068, loss = 0.003484
grad_step = 000069, loss = 0.003318
grad_step = 000070, loss = 0.003174
grad_step = 000071, loss = 0.003058
grad_step = 000072, loss = 0.002955
grad_step = 000073, loss = 0.002864
grad_step = 000074, loss = 0.002788
grad_step = 000075, loss = 0.002730
grad_step = 000076, loss = 0.002683
grad_step = 000077, loss = 0.002637
grad_step = 000078, loss = 0.002594
grad_step = 000079, loss = 0.002563
grad_step = 000080, loss = 0.002536
grad_step = 000081, loss = 0.002509
grad_step = 000082, loss = 0.002486
grad_step = 000083, loss = 0.002466
grad_step = 000084, loss = 0.002448
grad_step = 000085, loss = 0.002433
grad_step = 000086, loss = 0.002416
grad_step = 000087, loss = 0.002400
grad_step = 000088, loss = 0.002385
grad_step = 000089, loss = 0.002371
grad_step = 000090, loss = 0.002357
grad_step = 000091, loss = 0.002344
grad_step = 000092, loss = 0.002332
grad_step = 000093, loss = 0.002322
grad_step = 000094, loss = 0.002310
grad_step = 000095, loss = 0.002299
grad_step = 000096, loss = 0.002289
grad_step = 000097, loss = 0.002280
grad_step = 000098, loss = 0.002271
grad_step = 000099, loss = 0.002263
grad_step = 000100, loss = 0.002256
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002249
grad_step = 000102, loss = 0.002242
grad_step = 000103, loss = 0.002236
grad_step = 000104, loss = 0.002230
grad_step = 000105, loss = 0.002224
grad_step = 000106, loss = 0.002218
grad_step = 000107, loss = 0.002213
grad_step = 000108, loss = 0.002209
grad_step = 000109, loss = 0.002204
grad_step = 000110, loss = 0.002200
grad_step = 000111, loss = 0.002195
grad_step = 000112, loss = 0.002191
grad_step = 000113, loss = 0.002187
grad_step = 000114, loss = 0.002183
grad_step = 000115, loss = 0.002180
grad_step = 000116, loss = 0.002176
grad_step = 000117, loss = 0.002172
grad_step = 000118, loss = 0.002168
grad_step = 000119, loss = 0.002165
grad_step = 000120, loss = 0.002161
grad_step = 000121, loss = 0.002157
grad_step = 000122, loss = 0.002154
grad_step = 000123, loss = 0.002150
grad_step = 000124, loss = 0.002146
grad_step = 000125, loss = 0.002142
grad_step = 000126, loss = 0.002139
grad_step = 000127, loss = 0.002135
grad_step = 000128, loss = 0.002131
grad_step = 000129, loss = 0.002127
grad_step = 000130, loss = 0.002123
grad_step = 000131, loss = 0.002119
grad_step = 000132, loss = 0.002115
grad_step = 000133, loss = 0.002111
grad_step = 000134, loss = 0.002107
grad_step = 000135, loss = 0.002103
grad_step = 000136, loss = 0.002099
grad_step = 000137, loss = 0.002094
grad_step = 000138, loss = 0.002090
grad_step = 000139, loss = 0.002086
grad_step = 000140, loss = 0.002082
grad_step = 000141, loss = 0.002077
grad_step = 000142, loss = 0.002073
grad_step = 000143, loss = 0.002068
grad_step = 000144, loss = 0.002064
grad_step = 000145, loss = 0.002059
grad_step = 000146, loss = 0.002054
grad_step = 000147, loss = 0.002050
grad_step = 000148, loss = 0.002045
grad_step = 000149, loss = 0.002040
grad_step = 000150, loss = 0.002035
grad_step = 000151, loss = 0.002030
grad_step = 000152, loss = 0.002025
grad_step = 000153, loss = 0.002020
grad_step = 000154, loss = 0.002015
grad_step = 000155, loss = 0.002010
grad_step = 000156, loss = 0.002005
grad_step = 000157, loss = 0.002000
grad_step = 000158, loss = 0.001994
grad_step = 000159, loss = 0.001989
grad_step = 000160, loss = 0.001984
grad_step = 000161, loss = 0.001978
grad_step = 000162, loss = 0.001972
grad_step = 000163, loss = 0.001966
grad_step = 000164, loss = 0.001959
grad_step = 000165, loss = 0.001953
grad_step = 000166, loss = 0.001946
grad_step = 000167, loss = 0.001939
grad_step = 000168, loss = 0.001931
grad_step = 000169, loss = 0.001924
grad_step = 000170, loss = 0.001916
grad_step = 000171, loss = 0.001907
grad_step = 000172, loss = 0.001899
grad_step = 000173, loss = 0.001890
grad_step = 000174, loss = 0.001882
grad_step = 000175, loss = 0.001881
grad_step = 000176, loss = 0.001898
grad_step = 000177, loss = 0.001911
grad_step = 000178, loss = 0.001893
grad_step = 000179, loss = 0.001851
grad_step = 000180, loss = 0.001838
grad_step = 000181, loss = 0.001858
grad_step = 000182, loss = 0.001869
grad_step = 000183, loss = 0.001856
grad_step = 000184, loss = 0.001822
grad_step = 000185, loss = 0.001801
grad_step = 000186, loss = 0.001804
grad_step = 000187, loss = 0.001817
grad_step = 000188, loss = 0.001828
grad_step = 000189, loss = 0.001819
grad_step = 000190, loss = 0.001796
grad_step = 000191, loss = 0.001771
grad_step = 000192, loss = 0.001755
grad_step = 000193, loss = 0.001751
grad_step = 000194, loss = 0.001755
grad_step = 000195, loss = 0.001767
grad_step = 000196, loss = 0.001783
grad_step = 000197, loss = 0.001812
grad_step = 000198, loss = 0.001826
grad_step = 000199, loss = 0.001823
grad_step = 000200, loss = 0.001760
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001710
grad_step = 000202, loss = 0.001703
grad_step = 000203, loss = 0.001729
grad_step = 000204, loss = 0.001753
grad_step = 000205, loss = 0.001741
grad_step = 000206, loss = 0.001709
grad_step = 000207, loss = 0.001676
grad_step = 000208, loss = 0.001665
grad_step = 000209, loss = 0.001675
grad_step = 000210, loss = 0.001691
grad_step = 000211, loss = 0.001699
grad_step = 000212, loss = 0.001686
grad_step = 000213, loss = 0.001664
grad_step = 000214, loss = 0.001640
grad_step = 000215, loss = 0.001628
grad_step = 000216, loss = 0.001630
grad_step = 000217, loss = 0.001638
grad_step = 000218, loss = 0.001643
grad_step = 000219, loss = 0.001641
grad_step = 000220, loss = 0.001635
grad_step = 000221, loss = 0.001622
grad_step = 000222, loss = 0.001609
grad_step = 000223, loss = 0.001597
grad_step = 000224, loss = 0.001589
grad_step = 000225, loss = 0.001586
grad_step = 000226, loss = 0.001586
grad_step = 000227, loss = 0.001590
grad_step = 000228, loss = 0.001596
grad_step = 000229, loss = 0.001610
grad_step = 000230, loss = 0.001621
grad_step = 000231, loss = 0.001642
grad_step = 000232, loss = 0.001636
grad_step = 000233, loss = 0.001634
grad_step = 000234, loss = 0.001631
grad_step = 000235, loss = 0.001677
grad_step = 000236, loss = 0.001715
grad_step = 000237, loss = 0.001673
grad_step = 000238, loss = 0.001565
grad_step = 000239, loss = 0.001585
grad_step = 000240, loss = 0.001644
grad_step = 000241, loss = 0.001583
grad_step = 000242, loss = 0.001554
grad_step = 000243, loss = 0.001602
grad_step = 000244, loss = 0.001583
grad_step = 000245, loss = 0.001546
grad_step = 000246, loss = 0.001570
grad_step = 000247, loss = 0.001576
grad_step = 000248, loss = 0.001546
grad_step = 000249, loss = 0.001552
grad_step = 000250, loss = 0.001574
grad_step = 000251, loss = 0.001574
grad_step = 000252, loss = 0.001575
grad_step = 000253, loss = 0.001622
grad_step = 000254, loss = 0.001634
grad_step = 000255, loss = 0.001629
grad_step = 000256, loss = 0.001580
grad_step = 000257, loss = 0.001553
grad_step = 000258, loss = 0.001545
grad_step = 000259, loss = 0.001558
grad_step = 000260, loss = 0.001580
grad_step = 000261, loss = 0.001561
grad_step = 000262, loss = 0.001526
grad_step = 000263, loss = 0.001524
grad_step = 000264, loss = 0.001549
grad_step = 000265, loss = 0.001555
grad_step = 000266, loss = 0.001530
grad_step = 000267, loss = 0.001518
grad_step = 000268, loss = 0.001523
grad_step = 000269, loss = 0.001526
grad_step = 000270, loss = 0.001529
grad_step = 000271, loss = 0.001527
grad_step = 000272, loss = 0.001517
grad_step = 000273, loss = 0.001505
grad_step = 000274, loss = 0.001508
grad_step = 000275, loss = 0.001516
grad_step = 000276, loss = 0.001515
grad_step = 000277, loss = 0.001509
grad_step = 000278, loss = 0.001505
grad_step = 000279, loss = 0.001501
grad_step = 000280, loss = 0.001497
grad_step = 000281, loss = 0.001498
grad_step = 000282, loss = 0.001503
grad_step = 000283, loss = 0.001505
grad_step = 000284, loss = 0.001507
grad_step = 000285, loss = 0.001509
grad_step = 000286, loss = 0.001512
grad_step = 000287, loss = 0.001511
grad_step = 000288, loss = 0.001509
grad_step = 000289, loss = 0.001506
grad_step = 000290, loss = 0.001503
grad_step = 000291, loss = 0.001497
grad_step = 000292, loss = 0.001491
grad_step = 000293, loss = 0.001487
grad_step = 000294, loss = 0.001484
grad_step = 000295, loss = 0.001481
grad_step = 000296, loss = 0.001479
grad_step = 000297, loss = 0.001478
grad_step = 000298, loss = 0.001477
grad_step = 000299, loss = 0.001476
grad_step = 000300, loss = 0.001476
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001476
grad_step = 000302, loss = 0.001478
grad_step = 000303, loss = 0.001483
grad_step = 000304, loss = 0.001493
grad_step = 000305, loss = 0.001515
grad_step = 000306, loss = 0.001551
grad_step = 000307, loss = 0.001615
grad_step = 000308, loss = 0.001638
grad_step = 000309, loss = 0.001625
grad_step = 000310, loss = 0.001526
grad_step = 000311, loss = 0.001474
grad_step = 000312, loss = 0.001503
grad_step = 000313, loss = 0.001535
grad_step = 000314, loss = 0.001518
grad_step = 000315, loss = 0.001479
grad_step = 000316, loss = 0.001482
grad_step = 000317, loss = 0.001502
grad_step = 000318, loss = 0.001488
grad_step = 000319, loss = 0.001474
grad_step = 000320, loss = 0.001477
grad_step = 000321, loss = 0.001479
grad_step = 000322, loss = 0.001474
grad_step = 000323, loss = 0.001466
grad_step = 000324, loss = 0.001466
grad_step = 000325, loss = 0.001466
grad_step = 000326, loss = 0.001462
grad_step = 000327, loss = 0.001462
grad_step = 000328, loss = 0.001461
grad_step = 000329, loss = 0.001455
grad_step = 000330, loss = 0.001451
grad_step = 000331, loss = 0.001454
grad_step = 000332, loss = 0.001456
grad_step = 000333, loss = 0.001449
grad_step = 000334, loss = 0.001443
grad_step = 000335, loss = 0.001445
grad_step = 000336, loss = 0.001448
grad_step = 000337, loss = 0.001446
grad_step = 000338, loss = 0.001440
grad_step = 000339, loss = 0.001437
grad_step = 000340, loss = 0.001439
grad_step = 000341, loss = 0.001440
grad_step = 000342, loss = 0.001437
grad_step = 000343, loss = 0.001434
grad_step = 000344, loss = 0.001433
grad_step = 000345, loss = 0.001433
grad_step = 000346, loss = 0.001433
grad_step = 000347, loss = 0.001430
grad_step = 000348, loss = 0.001428
grad_step = 000349, loss = 0.001427
grad_step = 000350, loss = 0.001427
grad_step = 000351, loss = 0.001427
grad_step = 000352, loss = 0.001425
grad_step = 000353, loss = 0.001424
grad_step = 000354, loss = 0.001423
grad_step = 000355, loss = 0.001423
grad_step = 000356, loss = 0.001422
grad_step = 000357, loss = 0.001421
grad_step = 000358, loss = 0.001420
grad_step = 000359, loss = 0.001420
grad_step = 000360, loss = 0.001421
grad_step = 000361, loss = 0.001423
grad_step = 000362, loss = 0.001426
grad_step = 000363, loss = 0.001433
grad_step = 000364, loss = 0.001442
grad_step = 000365, loss = 0.001461
grad_step = 000366, loss = 0.001483
grad_step = 000367, loss = 0.001518
grad_step = 000368, loss = 0.001532
grad_step = 000369, loss = 0.001534
grad_step = 000370, loss = 0.001484
grad_step = 000371, loss = 0.001433
grad_step = 000372, loss = 0.001407
grad_step = 000373, loss = 0.001423
grad_step = 000374, loss = 0.001453
grad_step = 000375, loss = 0.001453
grad_step = 000376, loss = 0.001431
grad_step = 000377, loss = 0.001406
grad_step = 000378, loss = 0.001405
grad_step = 000379, loss = 0.001420
grad_step = 000380, loss = 0.001427
grad_step = 000381, loss = 0.001417
grad_step = 000382, loss = 0.001401
grad_step = 000383, loss = 0.001397
grad_step = 000384, loss = 0.001405
grad_step = 000385, loss = 0.001410
grad_step = 000386, loss = 0.001406
grad_step = 000387, loss = 0.001397
grad_step = 000388, loss = 0.001392
grad_step = 000389, loss = 0.001393
grad_step = 000390, loss = 0.001395
grad_step = 000391, loss = 0.001395
grad_step = 000392, loss = 0.001392
grad_step = 000393, loss = 0.001390
grad_step = 000394, loss = 0.001388
grad_step = 000395, loss = 0.001386
grad_step = 000396, loss = 0.001384
grad_step = 000397, loss = 0.001382
grad_step = 000398, loss = 0.001382
grad_step = 000399, loss = 0.001383
grad_step = 000400, loss = 0.001383
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001381
grad_step = 000402, loss = 0.001378
grad_step = 000403, loss = 0.001376
grad_step = 000404, loss = 0.001374
grad_step = 000405, loss = 0.001373
grad_step = 000406, loss = 0.001373
grad_step = 000407, loss = 0.001372
grad_step = 000408, loss = 0.001370
grad_step = 000409, loss = 0.001369
grad_step = 000410, loss = 0.001367
grad_step = 000411, loss = 0.001367
grad_step = 000412, loss = 0.001366
grad_step = 000413, loss = 0.001366
grad_step = 000414, loss = 0.001366
grad_step = 000415, loss = 0.001367
grad_step = 000416, loss = 0.001367
grad_step = 000417, loss = 0.001370
grad_step = 000418, loss = 0.001374
grad_step = 000419, loss = 0.001382
grad_step = 000420, loss = 0.001395
grad_step = 000421, loss = 0.001420
grad_step = 000422, loss = 0.001451
grad_step = 000423, loss = 0.001505
grad_step = 000424, loss = 0.001538
grad_step = 000425, loss = 0.001564
grad_step = 000426, loss = 0.001494
grad_step = 000427, loss = 0.001409
grad_step = 000428, loss = 0.001354
grad_step = 000429, loss = 0.001379
grad_step = 000430, loss = 0.001432
grad_step = 000431, loss = 0.001422
grad_step = 000432, loss = 0.001377
grad_step = 000433, loss = 0.001349
grad_step = 000434, loss = 0.001371
grad_step = 000435, loss = 0.001404
grad_step = 000436, loss = 0.001389
grad_step = 000437, loss = 0.001356
grad_step = 000438, loss = 0.001344
grad_step = 000439, loss = 0.001362
grad_step = 000440, loss = 0.001375
grad_step = 000441, loss = 0.001360
grad_step = 000442, loss = 0.001341
grad_step = 000443, loss = 0.001340
grad_step = 000444, loss = 0.001352
grad_step = 000445, loss = 0.001359
grad_step = 000446, loss = 0.001349
grad_step = 000447, loss = 0.001337
grad_step = 000448, loss = 0.001332
grad_step = 000449, loss = 0.001335
grad_step = 000450, loss = 0.001341
grad_step = 000451, loss = 0.001341
grad_step = 000452, loss = 0.001335
grad_step = 000453, loss = 0.001329
grad_step = 000454, loss = 0.001326
grad_step = 000455, loss = 0.001327
grad_step = 000456, loss = 0.001328
grad_step = 000457, loss = 0.001329
grad_step = 000458, loss = 0.001327
grad_step = 000459, loss = 0.001325
grad_step = 000460, loss = 0.001322
grad_step = 000461, loss = 0.001320
grad_step = 000462, loss = 0.001318
grad_step = 000463, loss = 0.001317
grad_step = 000464, loss = 0.001317
grad_step = 000465, loss = 0.001317
grad_step = 000466, loss = 0.001316
grad_step = 000467, loss = 0.001316
grad_step = 000468, loss = 0.001316
grad_step = 000469, loss = 0.001316
grad_step = 000470, loss = 0.001315
grad_step = 000471, loss = 0.001314
grad_step = 000472, loss = 0.001313
grad_step = 000473, loss = 0.001311
grad_step = 000474, loss = 0.001310
grad_step = 000475, loss = 0.001308
grad_step = 000476, loss = 0.001307
grad_step = 000477, loss = 0.001307
grad_step = 000478, loss = 0.001307
grad_step = 000479, loss = 0.001307
grad_step = 000480, loss = 0.001309
grad_step = 000481, loss = 0.001310
grad_step = 000482, loss = 0.001315
grad_step = 000483, loss = 0.001320
grad_step = 000484, loss = 0.001329
grad_step = 000485, loss = 0.001339
grad_step = 000486, loss = 0.001356
grad_step = 000487, loss = 0.001366
grad_step = 000488, loss = 0.001382
grad_step = 000489, loss = 0.001377
grad_step = 000490, loss = 0.001370
grad_step = 000491, loss = 0.001347
grad_step = 000492, loss = 0.001333
grad_step = 000493, loss = 0.001323
grad_step = 000494, loss = 0.001321
grad_step = 000495, loss = 0.001326
grad_step = 000496, loss = 0.001321
grad_step = 000497, loss = 0.001313
grad_step = 000498, loss = 0.001298
grad_step = 000499, loss = 0.001290
grad_step = 000500, loss = 0.001293
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001301
Finished.

  #### Inference Need return ypred, ytrue ######################### 
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:15:13.064342
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.221091
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 23:15:13.070063
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    0.1256
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 23:15:13.078158
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.128027
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 23:15:13.083560
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.908543
metric_name                                             r2_score
Name: 3, dtype: object 

  


### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/armdn/'} 

  #### Setup Model   ############################################## 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_probability/python/distributions/mixture.py:154: Categorical.event_size (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-05-19.
Instructions for updating:
The `event_size` property is deprecated.  Use `num_categories` instead.  They have the same value, but `event_size` is misnamed.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
LSTM_1 (LSTM)                (None, 60, 300)           362400    
_________________________________________________________________
LSTM_2 (LSTM)                (None, 60, 200)           400800    
_________________________________________________________________
LSTM_3 (LSTM)                (None, 60, 24)            21600     
_________________________________________________________________
LSTM_4 (LSTM)                (None, 12)                1776      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                130       
_________________________________________________________________
mdn_1 (MDN)                  (None, 363)               3993      
=================================================================
Total params: 790,699
Trainable params: 790,699
Non-trainable params: 0
_________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f962b825198> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352943.8750
Epoch 2/10

1/1 [==============================] - 0s 108ms/step - loss: 253893.9375
Epoch 3/10

1/1 [==============================] - 0s 101ms/step - loss: 155614.6875
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 90272.8594
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 53051.5586
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 32823.5977
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 21631.1758
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 15128.4482
Epoch 9/10

1/1 [==============================] - 0s 113ms/step - loss: 11163.6309
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 8627.6553

  #### Inference Need return ypred, ytrue ######################### 
[[ 9.4905026e-02  8.6968451e+00  6.3254294e+00  6.6731262e+00
   7.9198041e+00  7.8543329e+00  6.7110128e+00  6.4856210e+00
   5.7426481e+00  7.2066464e+00  5.5725245e+00  6.9747386e+00
   7.6046958e+00  7.8929906e+00  5.8568921e+00  5.4936810e+00
   8.1439619e+00  7.6831040e+00  6.2493834e+00  8.0701199e+00
   6.0252008e+00  8.7396097e+00  7.8098164e+00  8.2410355e+00
   7.3914127e+00  7.0700159e+00  7.3597322e+00  8.5709400e+00
   6.5041943e+00  6.0182309e+00  6.6154695e+00  7.8185644e+00
   7.0119114e+00  6.0641060e+00  6.9978356e+00  7.6482325e+00
   5.1214671e+00  7.5545201e+00  4.3732438e+00  6.2187681e+00
   5.4381862e+00  7.1288700e+00  7.8845496e+00  6.0316086e+00
   5.5829110e+00  7.3226600e+00  8.3734798e+00  5.8406906e+00
   5.8505378e+00  7.5437880e+00  5.8095531e+00  6.5544086e+00
   6.9687662e+00  6.8865800e+00  7.2619615e+00  7.4157939e+00
   7.9215441e+00  6.7676163e+00  6.7851858e+00  5.9345222e+00
  -1.3790419e+00 -2.9542851e-01 -1.8717450e+00  4.6948463e-01
  -2.8306741e-01 -2.3493320e-01 -4.7058418e-02 -6.2360168e-03
   9.6431158e-02 -1.2314668e+00 -1.7999210e+00  9.3216598e-01
  -3.4976134e-01 -5.4360515e-01 -7.3397839e-01  1.5104628e-01
   4.8082608e-01 -8.3177495e-01 -6.6228062e-02  5.4705381e-02
  -1.1488271e-01  5.7890266e-01  4.1254258e-01 -2.8020185e-01
   5.1514888e-01  2.7321386e-01  7.0336401e-01  1.5227544e+00
  -2.1191096e+00 -3.0732176e-01  8.4599328e-01  1.4745163e+00
  -5.1479429e-01 -1.3664770e+00 -8.3820230e-01  1.0239878e+00
   2.0241982e-01  1.8525448e+00  1.5334470e+00 -9.3359947e-02
   5.1965106e-01 -3.9205259e-01 -3.1663072e-01 -1.2287853e+00
  -9.0907311e-01 -1.9234705e-01  1.1265476e+00  3.3099353e-03
   7.1667695e-01  1.6835344e+00  1.7266612e+00 -4.7974205e-01
  -1.4562324e+00  2.0174953e-01  1.0329027e+00  4.2056406e-01
  -5.3842515e-01 -1.6616087e+00  2.3222888e-01 -1.0301228e+00
  -5.5479503e-01  8.7592852e-01  1.3273908e+00  1.7721024e+00
   5.9267420e-01 -1.5178900e+00  7.6070392e-01  1.2071093e+00
  -7.8488040e-01 -1.7797217e+00 -4.0856221e-01  2.3915523e-01
   1.2316092e+00  1.6376195e+00  1.3374887e+00  9.0083736e-01
   3.5430813e-01 -9.3139398e-01 -9.9050999e-01 -1.7245489e-01
  -1.7581735e+00  3.4858286e-03 -5.3835303e-01 -9.1455626e-01
  -5.6415153e-01 -5.5271101e-01  1.2945106e+00  9.7641975e-01
  -9.6881628e-02  6.8037882e-02  5.3706318e-02 -5.3396082e-01
  -1.0609127e+00 -2.2648916e+00 -1.9578063e-01  1.1335033e+00
   1.9240411e-01 -2.2580232e-01 -3.8785756e-02 -4.6969205e-01
  -1.5139549e+00  1.7020206e+00 -6.7928141e-01  1.2767682e+00
  -5.9785283e-01  1.4919014e+00 -6.8554956e-01  1.6299963e-02
   4.9167201e-01 -8.3632153e-01 -2.0377269e+00  1.1060534e+00
  -1.0054772e+00  6.3828230e-01 -1.5232625e+00 -1.0676944e+00
   1.6439667e+00  1.9106290e+00 -6.6914451e-01  1.4548869e-01
   7.1086347e-02  7.7976303e+00  8.0409069e+00  7.8471913e+00
   7.9677920e+00  6.7928162e+00  5.8274012e+00  5.6118422e+00
   5.8975124e+00  7.2289119e+00  5.9901032e+00  7.6606669e+00
   7.9529138e+00  8.5880508e+00  8.3161325e+00  7.3084068e+00
   6.6781759e+00  5.3048987e+00  8.9827204e+00  8.3325443e+00
   7.3090119e+00  6.2256842e+00  5.6057782e+00  7.6250000e+00
   6.3264689e+00  8.0018663e+00  8.2626677e+00  8.5087442e+00
   8.6718397e+00  7.0910134e+00  7.4813805e+00  9.0600319e+00
   5.5490284e+00  7.7971735e+00  6.8088622e+00  8.4351959e+00
   5.8666444e+00  7.3081737e+00  6.3564124e+00  6.5414677e+00
   8.4062901e+00  8.5074005e+00  7.7765579e+00  8.2321301e+00
   6.6459465e+00  7.8346095e+00  6.7084155e+00  6.7781134e+00
   8.7071724e+00  8.3141670e+00  8.9698219e+00  8.1868095e+00
   7.7678785e+00  7.3845544e+00  6.8267732e+00  7.0928450e+00
   6.6385765e+00  8.4774170e+00  5.7902894e+00  8.1617832e+00
   6.2481046e-01  2.2706804e+00  4.7441328e-01  3.6714822e-01
   2.1450684e+00  1.2472731e+00  4.5794588e-01  6.1363500e-01
   1.0728891e+00  2.5068998e-01  2.2982073e+00  1.3783917e+00
   7.2087741e-01  1.7927444e-01  1.8556800e+00  1.4767802e-01
   4.2521250e-01  1.1351120e+00  4.7902143e-01  1.0979421e+00
   1.4245927e+00  9.9408984e-01  9.0925997e-01  6.8069518e-01
   3.7551779e-01  5.5718255e-01  2.2848406e+00  1.6432238e+00
   1.2840461e+00  1.5279486e+00  1.1216149e+00  7.5777906e-01
   1.0951240e+00  7.1268356e-01  1.5372334e+00  8.0636120e-01
   3.2760820e+00  5.3071636e-01  7.9610580e-01  4.4487488e-01
   8.7312490e-01  3.7264901e-01  5.2782309e-01  1.0042492e+00
   2.4438918e-01  1.3898857e+00  2.9126506e+00  7.3251265e-01
   1.0701647e+00  4.3173885e-01  7.0845866e-01  2.9376721e-01
   2.0779834e+00  4.2274320e-01  2.0951748e-01  8.1447709e-01
   1.4418604e+00  2.3679757e+00  4.6027875e-01  1.6071455e+00
   2.2531939e+00  1.8669640e+00  1.7198931e+00  6.7327911e-01
   1.9492362e+00  6.7692077e-01  1.8555299e+00  1.0318100e+00
   2.0045549e-01  2.8551924e-01  3.0208188e-01  1.2433401e+00
   4.7513425e-01  4.6758449e-01  2.6824892e-01  3.5964668e-01
   8.3866853e-01  2.2059362e+00  7.2580618e-01  3.9798522e-01
   1.4508992e-01  5.8847755e-01  6.5630031e-01  4.6328837e-01
   1.5784531e+00  8.2486904e-01  3.2445210e-01  2.1005446e-01
   9.1491562e-01  6.7489254e-01  1.4045637e+00  1.4677784e+00
   1.0481496e+00  2.1346951e+00  1.1780816e+00  8.0913264e-01
   3.6757553e-01  1.7498394e+00  2.7363545e-01  2.1899003e-01
   2.4789395e+00  2.6995409e-01  1.2801660e+00  1.7139304e+00
   3.7606335e-01  8.1404448e-01  1.8986086e+00  3.1005931e-01
   3.4463793e-01  1.2526300e+00  1.9609876e+00  1.5563705e+00
   6.0825270e-01  4.3484777e-01  1.7305064e+00  4.5479178e-01
   2.3785825e+00  1.1997181e+00  3.7914515e-01  5.0125408e-01
   1.0696014e+01 -9.8102465e+00 -2.2190745e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:15:22.095939
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.2513
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 23:15:22.100205
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9098.76
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 23:15:22.104034
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.9414
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 23:15:22.107770
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -813.858
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_fb/fb_prophet/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
Initial log joint probability = -192.039
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      99       9186.38     0.0272386        1207.2           1           1      123   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     199       10269.2     0.0242289       2566.31        0.89        0.89      233   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     299       10621.2     0.0237499       3262.95           1           1      343   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     399       10886.5     0.0339822       1343.14           1           1      459   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     499       11288.1    0.00255943       1266.79           1           1      580   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     599       11498.7     0.0166167       2146.51           1           1      698   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     699       11555.9     0.0104637       2039.91           1           1      812   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     799       11575.2    0.00955805       570.757           1           1      922   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     899       11630.7     0.0178715       1643.41      0.3435      0.3435     1036   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     999       11700.1      0.034504       2394.16           1           1     1146   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1099       11744.7   0.000237394       144.685           1           1     1258   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1199       11753.1    0.00188838       552.132      0.4814           1     1372   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1299         11758    0.00101299       262.652      0.7415      0.7415     1490   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1399         11761   0.000712302       157.258           1           1     1606   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1499       11781.3     0.0243264       931.457           1           1     1717   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1599       11791.1     0.0025484       550.483      0.7644      0.7644     1834   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1699       11797.7    0.00732868       810.153           1           1     1952   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1799       11802.5   0.000319611       98.1955     0.04871           1     2077   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1818       11803.2   5.97419e-05       246.505   3.588e-07       0.001     2142  LS failed, Hessian reset 
    1855       11803.6   0.000110613       144.447   1.529e-06       0.001     2225  LS failed, Hessian reset 
    1899       11804.3   0.000976631       305.295           1           1     2275   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    1999       11805.4   4.67236e-05       72.2243      0.9487      0.9487     2391   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2033       11806.1   1.47341e-05       111.754   8.766e-08       0.001     2480  LS failed, Hessian reset 
    2099       11806.6   9.53816e-05       108.311      0.9684      0.9684     2563   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2151       11806.8   3.32394e-05       152.834   3.931e-07       0.001     2668  LS failed, Hessian reset 
    2199         11807    0.00273479       216.444           1           1     2723   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2299       11810.9    0.00793685       550.165           1           1     2837   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2399       11818.9     0.0134452       377.542           1           1     2952   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2499       11824.9     0.0041384       130.511           1           1     3060   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2525       11826.5   2.36518e-05       102.803   6.403e-08       0.001     3158  LS failed, Hessian reset 
    2599       11827.9   0.000370724       186.394      0.4637      0.4637     3242   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2606         11828   1.70497e-05       123.589     7.9e-08       0.001     3292  LS failed, Hessian reset 
    2699       11829.1    0.00168243       332.201           1           1     3407   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2709       11829.2   1.92694e-05       146.345   1.034e-07       0.001     3461  LS failed, Hessian reset 
    2746       11829.4   1.61976e-05       125.824   9.572e-08       0.001     3551  LS failed, Hessian reset 
    2799       11829.5    0.00491161       122.515           1           1     3615   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2899       11830.6   0.000250007       100.524           1           1     3742   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    2999       11830.9    0.00236328       193.309           1           1     3889   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3099       11831.3   0.000309242       194.211      0.7059      0.7059     4015   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3199       11831.4    1.3396e-05       91.8042      0.9217      0.9217     4136   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3299       11831.6   0.000373334       77.3538      0.3184           1     4256   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3399       11831.8   0.000125272       64.7127           1           1     4379   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3499         11832     0.0010491       69.8273           1           1     4503   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
    3553       11832.1   1.09422e-05       89.3197   8.979e-08       0.001     4612  LS failed, Hessian reset 
    3584       11832.1   8.65844e-07       55.9367      0.4252      0.4252     4658   
Optimization terminated normally: 
  Convergence detected: relative gradient magnitude is below tolerance
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f95a41212e8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 23:15:37.627375
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 23:15:37.630624
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 23:15:37.633704
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 23:15:37.636725
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -18.2877
metric_name                                             r2_score
Name: 11, dtype: object 

  


### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/ 

                        date_run  ...            metric_name
0   2020-05-10 23:15:13.064342  ...    mean_absolute_error
1   2020-05-10 23:15:13.070063  ...     mean_squared_error
2   2020-05-10 23:15:13.078158  ...  median_absolute_error
3   2020-05-10 23:15:13.083560  ...               r2_score
4   2020-05-10 23:15:22.095939  ...    mean_absolute_error
5   2020-05-10 23:15:22.100205  ...     mean_squared_error
6   2020-05-10 23:15:22.104034  ...  median_absolute_error
7   2020-05-10 23:15:22.107770  ...               r2_score
8   2020-05-10 23:15:37.627375  ...    mean_absolute_error
9   2020-05-10 23:15:37.630624  ...     mean_squared_error
10  2020-05-10 23:15:37.633704  ...  median_absolute_error
11  2020-05-10 23:15:37.636725  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 118, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

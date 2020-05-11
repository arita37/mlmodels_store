
  test_cli /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_cli', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_cli 

  # Testing Command Line System   





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/390cac7ee9f7623cf7fa6990086c5bab3f62fb80', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '390cac7ee9f7623cf7fa6990086c5bab3f62fb80', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/390cac7ee9f7623cf7fa6990086c5bab3f62fb80

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/390cac7ee9f7623cf7fa6990086c5bab3f62fb80

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n']





 ************************************************************************************************************************
ml_models --do init  --path ztest/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
init

  Working Folder /home/runner/work/mlmodels/mlmodels 
creating User Path : /home/runner/.mlmodels/

  Config values in Path user {'model_trained': '/home/runner/work/mlmodels/mlmodels/model_trained/', 'dataset': '/home/runner/work/mlmodels/mlmodels/dataset/'} 

  Check Config in Path user /home/runner/work/mlmodels/mlmodels/model_trained/ 





 ************************************************************************************************************************
ml_models --do model_list  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
model_list

 model_uri from : /home/runner/work/mlmodels/mlmodels/mlmodels/ 

model_tf.temporal_fusion_google
model_tf.util
model_tf.__init__
model_tf.1_lstm
model_dev.__init__
model_sklearn.model_lightgbm
model_sklearn.__init__
model_sklearn.model_sklearn
model_tch.util_data
model_tch.pplm
model_tch.transformer_sentence
model_tch.__init__
model_tch.nbeats
model_tch.textcnn
model_tch.mlp
model_tchtorch_vae
model_tch.03_nbeats_dataloader
model_tch.torchhub
model_tch.util_transformer
model_tch.transformer_classifier
model_tch.matchzoo_models
model_gluon.util_autogluon
model_gluon.util
model_gluon.gluonts_model
model_gluon.__init__
model_gluon.fb_prophet
model_gluon.gluon_automl
model_flow.__init__
model_rank.__init__
model_keras.charcnn
model_keras.charcnn_zhang
model_keras.namentity_crm_bilstm_dataloader
model_keras.01_deepctr
model_keras.util
model_keras.__init__
model_keras.namentity_crm_bilstm
model_keras.nbeats
model_keras.textcnn
model_keras.keras_gan
model_keras.textcnn_dataloader
model_keras.02_cnn
model_keras.preprocess
model_keras.Autokeras
model_keras.armdn
model_keras.textvae





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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 510, in main
    fit_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 418, in fit_cli
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 514, in main
    predict_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 429, in predict_cli
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f25febd9208> 

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
 [-0.0106555  -0.05818192 -0.07680754 -0.01760823 -0.03907425 -0.06222925]
 [ 0.20186654 -0.13619773 -0.07668617  0.27276891  0.13286179  0.26739493]
 [ 0.01891759  0.24449591  0.01528816  0.12333574 -0.08248542  0.18375151]
 [ 0.10292143  0.30366623  0.25246277  0.07194746  0.00281575  0.27409878]
 [ 0.12714961 -0.14826113 -0.43365237  0.67718571  0.02237409  0.13794653]
 [ 0.50795746  0.51259035  0.09970567  0.36095271  0.05264869  0.44471547]
 [ 0.09740487 -0.18103352 -0.14269346  0.47749925  0.050718   -0.02929993]
 [ 0.58178496  0.19064038  0.17559813  0.19808589 -0.4580521   0.23033157]
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
{'loss': 0.5859628729522228, 'loss_history': []}

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
{'loss': 0.49480509757995605, 'loss_history': []}

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

  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f65a1a39240> 

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
 [ 0.17568572 -0.04141981  0.1822551   0.13739263 -0.13240345  0.05182284]
 [-0.00939387  0.19176058 -0.05220594 -0.128783    0.0242072  -0.0704467 ]
 [-0.08542785 -0.03306087  0.03811447  0.23132917 -0.06976372  0.22167386]
 [ 0.17380495  0.16921504 -0.14243282 -0.03448388  0.15881072 -0.11077256]
 [-0.05415474  0.57635903  0.4814983   0.0594335  -0.28062981  0.45885566]
 [ 0.53082043 -0.04378338  0.36993304  0.04421099 -0.21546146  0.3162269 ]
 [ 0.4413974   0.00395593 -0.04343803  0.60301787 -0.51867777  0.46262357]
 [ 0.33878472 -0.19690882  0.30961743 -0.15588498  0.34177074  1.00080252]
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
{'loss': 0.439925879240036, 'loss_history': []}

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
{'loss': 0.4233381748199463, 'loss_history': []}

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
[32m[I 2020-05-11 03:17:16,591][0m Finished trial#0 resulted in value: 17.05914282798767. Current best value is 17.05914282798767 with parameters: {'learning_rate': 0.05669483358006167, 'num_layers': 4, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-11 03:17:18,722][0m Finished trial#1 resulted in value: 13.381626605987549. Current best value is 13.381626605987549 with parameters: {'learning_rate': 0.03976924054637391, 'num_layers': 4, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-11 03:17:25,629][0m Finished trial#0 resulted in value: 0.29331207275390625. Current best value is 0.29331207275390625 with parameters: {'learning_rate': 0.002918730424844932, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-11 03:17:27,540][0m Finished trial#1 resulted in value: 0.3218710497021675. Current best value is 0.29331207275390625 with parameters: {'learning_rate': 0.002918730424844932, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff0508cdeb8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:17:43.656911
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 03:17:43.664361
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 03:17:43.668464
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 03:17:43.671503
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff03c8642e8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355210.0625
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 313684.3750
Epoch 3/10

1/1 [==============================] - 0s 93ms/step - loss: 216665.0312
Epoch 4/10

1/1 [==============================] - 0s 95ms/step - loss: 151377.9219
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 99365.9688
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 64755.7734
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 43277.3320
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 29896.2812
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 21216.4844
Epoch 10/10

1/1 [==============================] - 0s 92ms/step - loss: 15388.4424

  #### Inference Need return ypred, ytrue ######################### 
[[-2.65751451e-01  5.20218801e+00  4.72949791e+00  4.10669374e+00
   4.18877220e+00  5.50274897e+00  5.31445169e+00  5.19026756e+00
   3.96930075e+00  3.92579842e+00  4.17956209e+00  4.76249218e+00
   5.34396267e+00  5.76754951e+00  5.00606966e+00  5.22740746e+00
   4.22574902e+00  5.16153717e+00  4.38166046e+00  4.90248156e+00
   3.86477447e+00  4.98120642e+00  4.88607693e+00  4.42708635e+00
   3.98124170e+00  3.57066131e+00  4.60325861e+00  5.29591417e+00
   5.12393522e+00  5.60973644e+00  4.21237612e+00  4.48313379e+00
   4.71628857e+00  5.12196302e+00  4.58159161e+00  5.22909069e+00
   4.01557779e+00  5.78589630e+00  5.40900707e+00  4.67389727e+00
   5.12169552e+00  4.68337917e+00  5.71871853e+00  3.75557947e+00
   5.41624260e+00  4.73597050e+00  3.76156139e+00  5.69175625e+00
   4.48631144e+00  3.55980420e+00  4.83917665e+00  5.44031525e+00
   4.41919851e+00  3.93584585e+00  5.96642399e+00  5.37378979e+00
   4.98110342e+00  4.62402391e+00  3.96591663e+00  3.92069602e+00
   1.69897616e-01  1.37534022e-01 -1.42469335e+00  9.03070211e-01
   4.91193533e-02  1.30112529e+00 -5.17811596e-01  4.59436566e-01
  -4.79917407e-01  6.24174476e-01 -1.98681355e-01 -2.50723451e-01
   5.35668969e-01 -8.26744437e-02 -4.85733449e-02  5.52740753e-01
   5.98367214e-01 -1.64942217e+00 -1.21240008e+00  7.10235536e-01
  -3.98188353e-01  1.52299777e-01  7.72623122e-01 -7.47584224e-01
   2.15736210e-01  4.02929783e-01 -7.08156049e-01 -2.30356842e-01
   4.38829184e-01  8.86282921e-01 -2.02907562e-01 -9.99731421e-01
  -3.96811962e-01 -1.18109965e+00 -3.04122597e-01  2.48667330e-01
   1.06483340e+00  1.39592677e-01 -1.48680234e+00  1.14919424e-01
  -1.02539349e+00  7.22220540e-03  1.69653147e-01  9.43182945e-01
  -6.40327632e-02 -8.21360171e-01  5.02985716e-02  3.18467021e-02
  -3.92453402e-01  3.12797874e-02 -2.62320906e-01  3.11473787e-01
   8.39825034e-01 -1.37565112e+00  4.21571940e-01  1.01824129e+00
  -1.12622857e-01  8.85853291e-01 -2.77106225e-01 -4.54475820e-01
   3.53139043e-02 -3.86042595e-02  7.67522812e-01 -3.15136552e-01
   5.53374171e-01  3.22603434e-03  6.29238784e-01 -1.63450682e+00
   3.87904257e-01 -4.66562331e-01 -5.09950936e-01 -4.68600392e-02
  -1.16388655e+00  1.88235581e-01  1.33419096e-01  7.63062477e-01
  -4.69387919e-01  1.76529884e-02  5.96473575e-01  3.87521297e-01
   9.01848853e-01  1.58711982e+00  9.22358632e-02 -2.55110800e-01
  -7.87714720e-02 -5.82714915e-01  3.05373520e-01 -5.14891267e-01
  -6.57648742e-02 -4.49020118e-01  3.45541626e-01  6.79428995e-01
   8.51334214e-01  2.09796965e-01 -1.24800563e+00 -6.90637589e-01
   1.10621259e-01 -2.13206142e-01  2.92741954e-01 -1.68320942e+00
   5.82374811e-01  3.65466624e-01  6.34319782e-02  1.13948369e+00
  -3.35479826e-01 -1.23829055e+00 -1.14886498e+00  1.37814760e+00
   3.47011149e-01  3.32370251e-02 -3.43564451e-01 -6.82935178e-01
   9.47417736e-01 -9.84355211e-01  7.24568605e-01  6.56754494e-01
  -2.10539609e-01 -1.28241014e+00 -9.65222716e-02  3.81620884e-01
   6.11593127e-02  5.99501944e+00  5.25890255e+00  6.21963310e+00
   5.35091019e+00  5.18483210e+00  5.08623791e+00  5.75271320e+00
   5.66705179e+00  5.57645321e+00  5.38214302e+00  4.37467432e+00
   5.16984987e+00  6.64866400e+00  6.23289204e+00  4.62820768e+00
   6.17635489e+00  5.04567289e+00  5.45632029e+00  6.40636778e+00
   5.89523268e+00  5.32472706e+00  6.50320721e+00  6.07520771e+00
   5.52831745e+00  6.24689960e+00  5.97060490e+00  5.93368149e+00
   4.20414829e+00  5.61123133e+00  5.99414015e+00  6.59717703e+00
   5.85912657e+00  4.45381451e+00  4.90195274e+00  4.89656639e+00
   5.56956530e+00  5.72184181e+00  5.50552225e+00  5.44030094e+00
   5.23688459e+00  5.97562551e+00  5.24811220e+00  5.92783880e+00
   5.14254570e+00  6.30330133e+00  6.15178347e+00  6.29803610e+00
   6.63932037e+00  5.27106380e+00  4.70344830e+00  6.23281670e+00
   5.58568811e+00  6.48167038e+00  5.39378977e+00  5.59929132e+00
   4.83609056e+00  5.36218405e+00  5.35474014e+00  4.93918037e+00
   9.35576320e-01  1.45015514e+00  1.12305701e+00  2.01862383e+00
   1.64036763e+00  1.34779620e+00  6.56488180e-01  1.49653316e+00
   6.59742892e-01  1.12377310e+00  1.27654433e+00  4.40034389e-01
   1.76264286e+00  2.75086820e-01  1.19902241e+00  1.36492252e+00
   5.86494148e-01  3.93284440e-01  1.58168674e+00  5.49767017e-01
   3.87727022e-01  5.90433717e-01  6.84752524e-01  8.69813502e-01
   1.37079072e+00  3.69863927e-01  1.11931014e+00  5.67004859e-01
   1.36650789e+00  2.20444298e+00  1.81077957e+00  6.14652753e-01
   2.94732034e-01  1.53726459e-01  9.90229368e-01  3.47316384e-01
   6.12261236e-01  4.08962846e-01  1.46712041e+00  3.78972769e-01
   6.48575485e-01  7.59111226e-01  5.98673403e-01  1.26344872e+00
   1.53557873e+00  1.78127372e+00  4.26233053e-01  1.00191927e+00
   1.26548362e+00  5.40800452e-01  9.54762101e-01  8.18216920e-01
   9.59465206e-01  1.13067818e+00  6.45363986e-01  3.59461129e-01
   1.75725389e+00  8.99236321e-01  1.14387989e+00  1.89336109e+00
   9.42518651e-01  1.02410150e+00  4.65683341e-01  4.74434555e-01
   2.14174700e+00  1.57692742e+00  2.72520781e-01  1.65284073e+00
   6.41986132e-01  1.74448037e+00  7.22682357e-01  7.31889606e-01
   4.06794012e-01  1.59480691e+00  1.98907566e+00  1.01737916e+00
   5.11115372e-01  1.61390829e+00  1.44507122e+00  5.94806194e-01
   9.31864381e-01  5.91028214e-01  1.55903590e+00  7.31517196e-01
   1.32557917e+00  5.04570246e-01  1.13487077e+00  1.71835971e+00
   9.04441237e-01  7.40975559e-01  5.40017903e-01  3.65529001e-01
   1.92102051e+00  2.28495860e+00  1.15008473e+00  3.19544852e-01
   1.09668028e+00  1.85815918e+00  8.88661742e-01  1.93878448e+00
   6.22988641e-01  3.63801479e-01  1.95048976e+00  1.30250359e+00
   3.93090606e-01  1.43263173e+00  6.32170081e-01  8.22176814e-01
   6.74169779e-01  1.22922087e+00  1.63659549e+00  1.66371274e+00
   1.14565670e+00  1.16042602e+00  7.84388125e-01  1.42497504e+00
   1.76700997e+00  3.45355749e-01  7.95563221e-01  6.62168145e-01
   8.97190857e+00 -5.48086452e+00 -4.75727367e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:17:52.254289
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   97.2257
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 03:17:52.258061
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9473.03
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 03:17:52.261664
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   97.9567
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 03:17:52.264902
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -847.377
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140669447097312
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140666916821704
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140666916822208
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140666916822712
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140666916839664
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140666916840168

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff050bae518> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.538840
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.503879
grad_step = 000002, loss = 0.475797
grad_step = 000003, loss = 0.444723
grad_step = 000004, loss = 0.414135
grad_step = 000005, loss = 0.391353
grad_step = 000006, loss = 0.380657
grad_step = 000007, loss = 0.373198
grad_step = 000008, loss = 0.356978
grad_step = 000009, loss = 0.340775
grad_step = 000010, loss = 0.329057
grad_step = 000011, loss = 0.320630
grad_step = 000012, loss = 0.312266
grad_step = 000013, loss = 0.302084
grad_step = 000014, loss = 0.290266
grad_step = 000015, loss = 0.278204
grad_step = 000016, loss = 0.267608
grad_step = 000017, loss = 0.258981
grad_step = 000018, loss = 0.250850
grad_step = 000019, loss = 0.241629
grad_step = 000020, loss = 0.231709
grad_step = 000021, loss = 0.222210
grad_step = 000022, loss = 0.213642
grad_step = 000023, loss = 0.205027
grad_step = 000024, loss = 0.196153
grad_step = 000025, loss = 0.187068
grad_step = 000026, loss = 0.178171
grad_step = 000027, loss = 0.169694
grad_step = 000028, loss = 0.161764
grad_step = 000029, loss = 0.154390
grad_step = 000030, loss = 0.146968
grad_step = 000031, loss = 0.139319
grad_step = 000032, loss = 0.132139
grad_step = 000033, loss = 0.125579
grad_step = 000034, loss = 0.119142
grad_step = 000035, loss = 0.112724
grad_step = 000036, loss = 0.106274
grad_step = 000037, loss = 0.100014
grad_step = 000038, loss = 0.094222
grad_step = 000039, loss = 0.088801
grad_step = 000040, loss = 0.083422
grad_step = 000041, loss = 0.078132
grad_step = 000042, loss = 0.073116
grad_step = 000043, loss = 0.068305
grad_step = 000044, loss = 0.063724
grad_step = 000045, loss = 0.059234
grad_step = 000046, loss = 0.054950
grad_step = 000047, loss = 0.051014
grad_step = 000048, loss = 0.047245
grad_step = 000049, loss = 0.043597
grad_step = 000050, loss = 0.040162
grad_step = 000051, loss = 0.036928
grad_step = 000052, loss = 0.033915
grad_step = 000053, loss = 0.030999
grad_step = 000054, loss = 0.028280
grad_step = 000055, loss = 0.025808
grad_step = 000056, loss = 0.023463
grad_step = 000057, loss = 0.021266
grad_step = 000058, loss = 0.019218
grad_step = 000059, loss = 0.017365
grad_step = 000060, loss = 0.015602
grad_step = 000061, loss = 0.013995
grad_step = 000062, loss = 0.012543
grad_step = 000063, loss = 0.011207
grad_step = 000064, loss = 0.009969
grad_step = 000065, loss = 0.008871
grad_step = 000066, loss = 0.007898
grad_step = 000067, loss = 0.007030
grad_step = 000068, loss = 0.006273
grad_step = 000069, loss = 0.005610
grad_step = 000070, loss = 0.005025
grad_step = 000071, loss = 0.004543
grad_step = 000072, loss = 0.004135
grad_step = 000073, loss = 0.003798
grad_step = 000074, loss = 0.003537
grad_step = 000075, loss = 0.003395
grad_step = 000076, loss = 0.003490
grad_step = 000077, loss = 0.003729
grad_step = 000078, loss = 0.003478
grad_step = 000079, loss = 0.002804
grad_step = 000080, loss = 0.003053
grad_step = 000081, loss = 0.003198
grad_step = 000082, loss = 0.002667
grad_step = 000083, loss = 0.002872
grad_step = 000084, loss = 0.002916
grad_step = 000085, loss = 0.002568
grad_step = 000086, loss = 0.002825
grad_step = 000087, loss = 0.002688
grad_step = 000088, loss = 0.002547
grad_step = 000089, loss = 0.002729
grad_step = 000090, loss = 0.002505
grad_step = 000091, loss = 0.002555
grad_step = 000092, loss = 0.002573
grad_step = 000093, loss = 0.002413
grad_step = 000094, loss = 0.002516
grad_step = 000095, loss = 0.002410
grad_step = 000096, loss = 0.002378
grad_step = 000097, loss = 0.002418
grad_step = 000098, loss = 0.002303
grad_step = 000099, loss = 0.002345
grad_step = 000100, loss = 0.002303
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002251
grad_step = 000102, loss = 0.002286
grad_step = 000103, loss = 0.002216
grad_step = 000104, loss = 0.002222
grad_step = 000105, loss = 0.002218
grad_step = 000106, loss = 0.002170
grad_step = 000107, loss = 0.002193
grad_step = 000108, loss = 0.002164
grad_step = 000109, loss = 0.002151
grad_step = 000110, loss = 0.002164
grad_step = 000111, loss = 0.002133
grad_step = 000112, loss = 0.002140
grad_step = 000113, loss = 0.002138
grad_step = 000114, loss = 0.002118
grad_step = 000115, loss = 0.002130
grad_step = 000116, loss = 0.002119
grad_step = 000117, loss = 0.002110
grad_step = 000118, loss = 0.002119
grad_step = 000119, loss = 0.002106
grad_step = 000120, loss = 0.002104
grad_step = 000121, loss = 0.002108
grad_step = 000122, loss = 0.002097
grad_step = 000123, loss = 0.002097
grad_step = 000124, loss = 0.002097
grad_step = 000125, loss = 0.002088
grad_step = 000126, loss = 0.002088
grad_step = 000127, loss = 0.002087
grad_step = 000128, loss = 0.002079
grad_step = 000129, loss = 0.002078
grad_step = 000130, loss = 0.002076
grad_step = 000131, loss = 0.002070
grad_step = 000132, loss = 0.002068
grad_step = 000133, loss = 0.002066
grad_step = 000134, loss = 0.002061
grad_step = 000135, loss = 0.002058
grad_step = 000136, loss = 0.002057
grad_step = 000137, loss = 0.002052
grad_step = 000138, loss = 0.002049
grad_step = 000139, loss = 0.002047
grad_step = 000140, loss = 0.002044
grad_step = 000141, loss = 0.002040
grad_step = 000142, loss = 0.002038
grad_step = 000143, loss = 0.002036
grad_step = 000144, loss = 0.002033
grad_step = 000145, loss = 0.002030
grad_step = 000146, loss = 0.002028
grad_step = 000147, loss = 0.002025
grad_step = 000148, loss = 0.002022
grad_step = 000149, loss = 0.002019
grad_step = 000150, loss = 0.002017
grad_step = 000151, loss = 0.002014
grad_step = 000152, loss = 0.002011
grad_step = 000153, loss = 0.002008
grad_step = 000154, loss = 0.002006
grad_step = 000155, loss = 0.002003
grad_step = 000156, loss = 0.002000
grad_step = 000157, loss = 0.001997
grad_step = 000158, loss = 0.001994
grad_step = 000159, loss = 0.001991
grad_step = 000160, loss = 0.001988
grad_step = 000161, loss = 0.001985
grad_step = 000162, loss = 0.001982
grad_step = 000163, loss = 0.001978
grad_step = 000164, loss = 0.001975
grad_step = 000165, loss = 0.001972
grad_step = 000166, loss = 0.001968
grad_step = 000167, loss = 0.001965
grad_step = 000168, loss = 0.001961
grad_step = 000169, loss = 0.001957
grad_step = 000170, loss = 0.001954
grad_step = 000171, loss = 0.001950
grad_step = 000172, loss = 0.001948
grad_step = 000173, loss = 0.001947
grad_step = 000174, loss = 0.001949
grad_step = 000175, loss = 0.001956
grad_step = 000176, loss = 0.001978
grad_step = 000177, loss = 0.002024
grad_step = 000178, loss = 0.002112
grad_step = 000179, loss = 0.002218
grad_step = 000180, loss = 0.002283
grad_step = 000181, loss = 0.002186
grad_step = 000182, loss = 0.001995
grad_step = 000183, loss = 0.001907
grad_step = 000184, loss = 0.001988
grad_step = 000185, loss = 0.002094
grad_step = 000186, loss = 0.002068
grad_step = 000187, loss = 0.001946
grad_step = 000188, loss = 0.001886
grad_step = 000189, loss = 0.001939
grad_step = 000190, loss = 0.002003
grad_step = 000191, loss = 0.001975
grad_step = 000192, loss = 0.001896
grad_step = 000193, loss = 0.001864
grad_step = 000194, loss = 0.001901
grad_step = 000195, loss = 0.001938
grad_step = 000196, loss = 0.001915
grad_step = 000197, loss = 0.001864
grad_step = 000198, loss = 0.001838
grad_step = 000199, loss = 0.001854
grad_step = 000200, loss = 0.001878
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001874
grad_step = 000202, loss = 0.001849
grad_step = 000203, loss = 0.001824
grad_step = 000204, loss = 0.001818
grad_step = 000205, loss = 0.001821
grad_step = 000206, loss = 0.001821
grad_step = 000207, loss = 0.001808
grad_step = 000208, loss = 0.001791
grad_step = 000209, loss = 0.001778
grad_step = 000210, loss = 0.001789
grad_step = 000211, loss = 0.001861
grad_step = 000212, loss = 0.002042
grad_step = 000213, loss = 0.001964
grad_step = 000214, loss = 0.001813
grad_step = 000215, loss = 0.001816
grad_step = 000216, loss = 0.001861
grad_step = 000217, loss = 0.001780
grad_step = 000218, loss = 0.001803
grad_step = 000219, loss = 0.001828
grad_step = 000220, loss = 0.001770
grad_step = 000221, loss = 0.001787
grad_step = 000222, loss = 0.001797
grad_step = 000223, loss = 0.001750
grad_step = 000224, loss = 0.001755
grad_step = 000225, loss = 0.001772
grad_step = 000226, loss = 0.001744
grad_step = 000227, loss = 0.001734
grad_step = 000228, loss = 0.001756
grad_step = 000229, loss = 0.001748
grad_step = 000230, loss = 0.001727
grad_step = 000231, loss = 0.001737
grad_step = 000232, loss = 0.001751
grad_step = 000233, loss = 0.001740
grad_step = 000234, loss = 0.001733
grad_step = 000235, loss = 0.001752
grad_step = 000236, loss = 0.001787
grad_step = 000237, loss = 0.001837
grad_step = 000238, loss = 0.001862
grad_step = 000239, loss = 0.001913
grad_step = 000240, loss = 0.001923
grad_step = 000241, loss = 0.001884
grad_step = 000242, loss = 0.001800
grad_step = 000243, loss = 0.001713
grad_step = 000244, loss = 0.001693
grad_step = 000245, loss = 0.001733
grad_step = 000246, loss = 0.001779
grad_step = 000247, loss = 0.001776
grad_step = 000248, loss = 0.001723
grad_step = 000249, loss = 0.001674
grad_step = 000250, loss = 0.001671
grad_step = 000251, loss = 0.001711
grad_step = 000252, loss = 0.001754
grad_step = 000253, loss = 0.001804
grad_step = 000254, loss = 0.001835
grad_step = 000255, loss = 0.001887
grad_step = 000256, loss = 0.001792
grad_step = 000257, loss = 0.001716
grad_step = 000258, loss = 0.001721
grad_step = 000259, loss = 0.001736
grad_step = 000260, loss = 0.001711
grad_step = 000261, loss = 0.001661
grad_step = 000262, loss = 0.001682
grad_step = 000263, loss = 0.001732
grad_step = 000264, loss = 0.001688
grad_step = 000265, loss = 0.001644
grad_step = 000266, loss = 0.001662
grad_step = 000267, loss = 0.001681
grad_step = 000268, loss = 0.001667
grad_step = 000269, loss = 0.001642
grad_step = 000270, loss = 0.001640
grad_step = 000271, loss = 0.001652
grad_step = 000272, loss = 0.001641
grad_step = 000273, loss = 0.001626
grad_step = 000274, loss = 0.001623
grad_step = 000275, loss = 0.001629
grad_step = 000276, loss = 0.001636
grad_step = 000277, loss = 0.001634
grad_step = 000278, loss = 0.001625
grad_step = 000279, loss = 0.001615
grad_step = 000280, loss = 0.001605
grad_step = 000281, loss = 0.001601
grad_step = 000282, loss = 0.001599
grad_step = 000283, loss = 0.001599
grad_step = 000284, loss = 0.001600
grad_step = 000285, loss = 0.001601
grad_step = 000286, loss = 0.001605
grad_step = 000287, loss = 0.001613
grad_step = 000288, loss = 0.001622
grad_step = 000289, loss = 0.001627
grad_step = 000290, loss = 0.001620
grad_step = 000291, loss = 0.001607
grad_step = 000292, loss = 0.001589
grad_step = 000293, loss = 0.001572
grad_step = 000294, loss = 0.001563
grad_step = 000295, loss = 0.001565
grad_step = 000296, loss = 0.001572
grad_step = 000297, loss = 0.001579
grad_step = 000298, loss = 0.001578
grad_step = 000299, loss = 0.001568
grad_step = 000300, loss = 0.001554
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001548
grad_step = 000302, loss = 0.001552
grad_step = 000303, loss = 0.001567
grad_step = 000304, loss = 0.001587
grad_step = 000305, loss = 0.001615
grad_step = 000306, loss = 0.001639
grad_step = 000307, loss = 0.001669
grad_step = 000308, loss = 0.001680
grad_step = 000309, loss = 0.001689
grad_step = 000310, loss = 0.001767
grad_step = 000311, loss = 0.001882
grad_step = 000312, loss = 0.001997
grad_step = 000313, loss = 0.001620
grad_step = 000314, loss = 0.001502
grad_step = 000315, loss = 0.001708
grad_step = 000316, loss = 0.001697
grad_step = 000317, loss = 0.001503
grad_step = 000318, loss = 0.001521
grad_step = 000319, loss = 0.001619
grad_step = 000320, loss = 0.001505
grad_step = 000321, loss = 0.001480
grad_step = 000322, loss = 0.001552
grad_step = 000323, loss = 0.001476
grad_step = 000324, loss = 0.001448
grad_step = 000325, loss = 0.001492
grad_step = 000326, loss = 0.001472
grad_step = 000327, loss = 0.001424
grad_step = 000328, loss = 0.001427
grad_step = 000329, loss = 0.001456
grad_step = 000330, loss = 0.001443
grad_step = 000331, loss = 0.001408
grad_step = 000332, loss = 0.001388
grad_step = 000333, loss = 0.001397
grad_step = 000334, loss = 0.001411
grad_step = 000335, loss = 0.001427
grad_step = 000336, loss = 0.001450
grad_step = 000337, loss = 0.001425
grad_step = 000338, loss = 0.001395
grad_step = 000339, loss = 0.001370
grad_step = 000340, loss = 0.001364
grad_step = 000341, loss = 0.001380
grad_step = 000342, loss = 0.001417
grad_step = 000343, loss = 0.001464
grad_step = 000344, loss = 0.001455
grad_step = 000345, loss = 0.001421
grad_step = 000346, loss = 0.001360
grad_step = 000347, loss = 0.001348
grad_step = 000348, loss = 0.001393
grad_step = 000349, loss = 0.001438
grad_step = 000350, loss = 0.001416
grad_step = 000351, loss = 0.001362
grad_step = 000352, loss = 0.001333
grad_step = 000353, loss = 0.001355
grad_step = 000354, loss = 0.001385
grad_step = 000355, loss = 0.001369
grad_step = 000356, loss = 0.001332
grad_step = 000357, loss = 0.001319
grad_step = 000358, loss = 0.001341
grad_step = 000359, loss = 0.001360
grad_step = 000360, loss = 0.001344
grad_step = 000361, loss = 0.001319
grad_step = 000362, loss = 0.001302
grad_step = 000363, loss = 0.001310
grad_step = 000364, loss = 0.001326
grad_step = 000365, loss = 0.001331
grad_step = 000366, loss = 0.001324
grad_step = 000367, loss = 0.001305
grad_step = 000368, loss = 0.001291
grad_step = 000369, loss = 0.001284
grad_step = 000370, loss = 0.001286
grad_step = 000371, loss = 0.001294
grad_step = 000372, loss = 0.001313
grad_step = 000373, loss = 0.001340
grad_step = 000374, loss = 0.001374
grad_step = 000375, loss = 0.001399
grad_step = 000376, loss = 0.001427
grad_step = 000377, loss = 0.001419
grad_step = 000378, loss = 0.001414
grad_step = 000379, loss = 0.001421
grad_step = 000380, loss = 0.001444
grad_step = 000381, loss = 0.001466
grad_step = 000382, loss = 0.001382
grad_step = 000383, loss = 0.001276
grad_step = 000384, loss = 0.001240
grad_step = 000385, loss = 0.001282
grad_step = 000386, loss = 0.001322
grad_step = 000387, loss = 0.001298
grad_step = 000388, loss = 0.001248
grad_step = 000389, loss = 0.001236
grad_step = 000390, loss = 0.001261
grad_step = 000391, loss = 0.001272
grad_step = 000392, loss = 0.001251
grad_step = 000393, loss = 0.001216
grad_step = 000394, loss = 0.001211
grad_step = 000395, loss = 0.001231
grad_step = 000396, loss = 0.001243
grad_step = 000397, loss = 0.001233
grad_step = 000398, loss = 0.001211
grad_step = 000399, loss = 0.001198
grad_step = 000400, loss = 0.001199
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001208
grad_step = 000402, loss = 0.001215
grad_step = 000403, loss = 0.001214
grad_step = 000404, loss = 0.001205
grad_step = 000405, loss = 0.001194
grad_step = 000406, loss = 0.001183
grad_step = 000407, loss = 0.001175
grad_step = 000408, loss = 0.001172
grad_step = 000409, loss = 0.001172
grad_step = 000410, loss = 0.001175
grad_step = 000411, loss = 0.001181
grad_step = 000412, loss = 0.001189
grad_step = 000413, loss = 0.001201
grad_step = 000414, loss = 0.001216
grad_step = 000415, loss = 0.001231
grad_step = 000416, loss = 0.001239
grad_step = 000417, loss = 0.001230
grad_step = 000418, loss = 0.001206
grad_step = 000419, loss = 0.001175
grad_step = 000420, loss = 0.001152
grad_step = 000421, loss = 0.001145
grad_step = 000422, loss = 0.001153
grad_step = 000423, loss = 0.001167
grad_step = 000424, loss = 0.001178
grad_step = 000425, loss = 0.001178
grad_step = 000426, loss = 0.001167
grad_step = 000427, loss = 0.001149
grad_step = 000428, loss = 0.001134
grad_step = 000429, loss = 0.001129
grad_step = 000430, loss = 0.001132
grad_step = 000431, loss = 0.001138
grad_step = 000432, loss = 0.001145
grad_step = 000433, loss = 0.001151
grad_step = 000434, loss = 0.001154
grad_step = 000435, loss = 0.001155
grad_step = 000436, loss = 0.001152
grad_step = 000437, loss = 0.001146
grad_step = 000438, loss = 0.001137
grad_step = 000439, loss = 0.001126
grad_step = 000440, loss = 0.001116
grad_step = 000441, loss = 0.001112
grad_step = 000442, loss = 0.001124
grad_step = 000443, loss = 0.001154
grad_step = 000444, loss = 0.001204
grad_step = 000445, loss = 0.001279
grad_step = 000446, loss = 0.001368
grad_step = 000447, loss = 0.001444
grad_step = 000448, loss = 0.001466
grad_step = 000449, loss = 0.001369
grad_step = 000450, loss = 0.001247
grad_step = 000451, loss = 0.001144
grad_step = 000452, loss = 0.001152
grad_step = 000453, loss = 0.001242
grad_step = 000454, loss = 0.001349
grad_step = 000455, loss = 0.001306
grad_step = 000456, loss = 0.001214
grad_step = 000457, loss = 0.001122
grad_step = 000458, loss = 0.001129
grad_step = 000459, loss = 0.001186
grad_step = 000460, loss = 0.001181
grad_step = 000461, loss = 0.001111
grad_step = 000462, loss = 0.001067
grad_step = 000463, loss = 0.001103
grad_step = 000464, loss = 0.001147
grad_step = 000465, loss = 0.001123
grad_step = 000466, loss = 0.001074
grad_step = 000467, loss = 0.001066
grad_step = 000468, loss = 0.001094
grad_step = 000469, loss = 0.001112
grad_step = 000470, loss = 0.001090
grad_step = 000471, loss = 0.001065
grad_step = 000472, loss = 0.001063
grad_step = 000473, loss = 0.001080
grad_step = 000474, loss = 0.001083
grad_step = 000475, loss = 0.001069
grad_step = 000476, loss = 0.001048
grad_step = 000477, loss = 0.001042
grad_step = 000478, loss = 0.001049
grad_step = 000479, loss = 0.001055
grad_step = 000480, loss = 0.001047
grad_step = 000481, loss = 0.001034
grad_step = 000482, loss = 0.001027
grad_step = 000483, loss = 0.001030
grad_step = 000484, loss = 0.001035
grad_step = 000485, loss = 0.001035
grad_step = 000486, loss = 0.001028
grad_step = 000487, loss = 0.001023
grad_step = 000488, loss = 0.001024
grad_step = 000489, loss = 0.001034
grad_step = 000490, loss = 0.001056
grad_step = 000491, loss = 0.001113
grad_step = 000492, loss = 0.001200
grad_step = 000493, loss = 0.001317
grad_step = 000494, loss = 0.001303
grad_step = 000495, loss = 0.001221
grad_step = 000496, loss = 0.001083
grad_step = 000497, loss = 0.001058
grad_step = 000498, loss = 0.001069
grad_step = 000499, loss = 0.001031
grad_step = 000500, loss = 0.001034
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001069
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

  date_run                              2020-05-11 03:18:11.039289
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.203013
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 03:18:11.045298
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.093515
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 03:18:11.052801
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.128173
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 03:18:11.058318
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.420994
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
0   2020-05-11 03:17:43.656911  ...    mean_absolute_error
1   2020-05-11 03:17:43.664361  ...     mean_squared_error
2   2020-05-11 03:17:43.668464  ...  median_absolute_error
3   2020-05-11 03:17:43.671503  ...               r2_score
4   2020-05-11 03:17:52.254289  ...    mean_absolute_error
5   2020-05-11 03:17:52.258061  ...     mean_squared_error
6   2020-05-11 03:17:52.261664  ...  median_absolute_error
7   2020-05-11 03:17:52.264902  ...               r2_score
8   2020-05-11 03:18:11.039289  ...    mean_absolute_error
9   2020-05-11 03:18:11.045298  ...     mean_squared_error
10  2020-05-11 03:18:11.052801  ...  median_absolute_error
11  2020-05-11 03:18:11.058318  ...               r2_score

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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140395798823320
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140394519546456
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140394519546960
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140394519547464
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140394519666816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140394519667320

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb071b7fef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.460499
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.427390
grad_step = 000002, loss = 0.402659
grad_step = 000003, loss = 0.377620
grad_step = 000004, loss = 0.350656
grad_step = 000005, loss = 0.323678
grad_step = 000006, loss = 0.302863
grad_step = 000007, loss = 0.286122
grad_step = 000008, loss = 0.267774
grad_step = 000009, loss = 0.250861
grad_step = 000010, loss = 0.229103
grad_step = 000011, loss = 0.213279
grad_step = 000012, loss = 0.202668
grad_step = 000013, loss = 0.192850
grad_step = 000014, loss = 0.184407
grad_step = 000015, loss = 0.176273
grad_step = 000016, loss = 0.167171
grad_step = 000017, loss = 0.157434
grad_step = 000018, loss = 0.147897
grad_step = 000019, loss = 0.138322
grad_step = 000020, loss = 0.129066
grad_step = 000021, loss = 0.121145
grad_step = 000022, loss = 0.114078
grad_step = 000023, loss = 0.105476
grad_step = 000024, loss = 0.096667
grad_step = 000025, loss = 0.088928
grad_step = 000026, loss = 0.081864
grad_step = 000027, loss = 0.075221
grad_step = 000028, loss = 0.069696
grad_step = 000029, loss = 0.065187
grad_step = 000030, loss = 0.060793
grad_step = 000031, loss = 0.055713
grad_step = 000032, loss = 0.050370
grad_step = 000033, loss = 0.045721
grad_step = 000034, loss = 0.041727
grad_step = 000035, loss = 0.038036
grad_step = 000036, loss = 0.034758
grad_step = 000037, loss = 0.031928
grad_step = 000038, loss = 0.029231
grad_step = 000039, loss = 0.026521
grad_step = 000040, loss = 0.023967
grad_step = 000041, loss = 0.021664
grad_step = 000042, loss = 0.019465
grad_step = 000043, loss = 0.017296
grad_step = 000044, loss = 0.015358
grad_step = 000045, loss = 0.013755
grad_step = 000046, loss = 0.012386
grad_step = 000047, loss = 0.011199
grad_step = 000048, loss = 0.010168
grad_step = 000049, loss = 0.009165
grad_step = 000050, loss = 0.008176
grad_step = 000051, loss = 0.007347
grad_step = 000052, loss = 0.006692
grad_step = 000053, loss = 0.006062
grad_step = 000054, loss = 0.005447
grad_step = 000055, loss = 0.004961
grad_step = 000056, loss = 0.004586
grad_step = 000057, loss = 0.004238
grad_step = 000058, loss = 0.003916
grad_step = 000059, loss = 0.003640
grad_step = 000060, loss = 0.003392
grad_step = 000061, loss = 0.003191
grad_step = 000062, loss = 0.003066
grad_step = 000063, loss = 0.002966
grad_step = 000064, loss = 0.002859
grad_step = 000065, loss = 0.002762
grad_step = 000066, loss = 0.002690
grad_step = 000067, loss = 0.002630
grad_step = 000068, loss = 0.002583
grad_step = 000069, loss = 0.002548
grad_step = 000070, loss = 0.002519
grad_step = 000071, loss = 0.002502
grad_step = 000072, loss = 0.002496
grad_step = 000073, loss = 0.002480
grad_step = 000074, loss = 0.002456
grad_step = 000075, loss = 0.002442
grad_step = 000076, loss = 0.002435
grad_step = 000077, loss = 0.002423
grad_step = 000078, loss = 0.002412
grad_step = 000079, loss = 0.002403
grad_step = 000080, loss = 0.002392
grad_step = 000081, loss = 0.002383
grad_step = 000082, loss = 0.002375
grad_step = 000083, loss = 0.002359
grad_step = 000084, loss = 0.002342
grad_step = 000085, loss = 0.002328
grad_step = 000086, loss = 0.002316
grad_step = 000087, loss = 0.002302
grad_step = 000088, loss = 0.002289
grad_step = 000089, loss = 0.002276
grad_step = 000090, loss = 0.002264
grad_step = 000091, loss = 0.002253
grad_step = 000092, loss = 0.002241
grad_step = 000093, loss = 0.002229
grad_step = 000094, loss = 0.002219
grad_step = 000095, loss = 0.002209
grad_step = 000096, loss = 0.002198
grad_step = 000097, loss = 0.002188
grad_step = 000098, loss = 0.002179
grad_step = 000099, loss = 0.002171
grad_step = 000100, loss = 0.002164
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002156
grad_step = 000102, loss = 0.002149
grad_step = 000103, loss = 0.002142
grad_step = 000104, loss = 0.002135
grad_step = 000105, loss = 0.002128
grad_step = 000106, loss = 0.002122
grad_step = 000107, loss = 0.002117
grad_step = 000108, loss = 0.002111
grad_step = 000109, loss = 0.002106
grad_step = 000110, loss = 0.002100
grad_step = 000111, loss = 0.002095
grad_step = 000112, loss = 0.002090
grad_step = 000113, loss = 0.002085
grad_step = 000114, loss = 0.002080
grad_step = 000115, loss = 0.002075
grad_step = 000116, loss = 0.002070
grad_step = 000117, loss = 0.002065
grad_step = 000118, loss = 0.002060
grad_step = 000119, loss = 0.002055
grad_step = 000120, loss = 0.002050
grad_step = 000121, loss = 0.002045
grad_step = 000122, loss = 0.002040
grad_step = 000123, loss = 0.002035
grad_step = 000124, loss = 0.002031
grad_step = 000125, loss = 0.002027
grad_step = 000126, loss = 0.002025
grad_step = 000127, loss = 0.002034
grad_step = 000128, loss = 0.002033
grad_step = 000129, loss = 0.002014
grad_step = 000130, loss = 0.002002
grad_step = 000131, loss = 0.002008
grad_step = 000132, loss = 0.002008
grad_step = 000133, loss = 0.001992
grad_step = 000134, loss = 0.001982
grad_step = 000135, loss = 0.001985
grad_step = 000136, loss = 0.001985
grad_step = 000137, loss = 0.001973
grad_step = 000138, loss = 0.001962
grad_step = 000139, loss = 0.001961
grad_step = 000140, loss = 0.001962
grad_step = 000141, loss = 0.001957
grad_step = 000142, loss = 0.001946
grad_step = 000143, loss = 0.001937
grad_step = 000144, loss = 0.001934
grad_step = 000145, loss = 0.001934
grad_step = 000146, loss = 0.001932
grad_step = 000147, loss = 0.001927
grad_step = 000148, loss = 0.001921
grad_step = 000149, loss = 0.001913
grad_step = 000150, loss = 0.001904
grad_step = 000151, loss = 0.001897
grad_step = 000152, loss = 0.001891
grad_step = 000153, loss = 0.001886
grad_step = 000154, loss = 0.001882
grad_step = 000155, loss = 0.001883
grad_step = 000156, loss = 0.001895
grad_step = 000157, loss = 0.001934
grad_step = 000158, loss = 0.002017
grad_step = 000159, loss = 0.002046
grad_step = 000160, loss = 0.001980
grad_step = 000161, loss = 0.001866
grad_step = 000162, loss = 0.001875
grad_step = 000163, loss = 0.001945
grad_step = 000164, loss = 0.001937
grad_step = 000165, loss = 0.001870
grad_step = 000166, loss = 0.001827
grad_step = 000167, loss = 0.001869
grad_step = 000168, loss = 0.001897
grad_step = 000169, loss = 0.001851
grad_step = 000170, loss = 0.001819
grad_step = 000171, loss = 0.001818
grad_step = 000172, loss = 0.001839
grad_step = 000173, loss = 0.001850
grad_step = 000174, loss = 0.001814
grad_step = 000175, loss = 0.001790
grad_step = 000176, loss = 0.001789
grad_step = 000177, loss = 0.001793
grad_step = 000178, loss = 0.001810
grad_step = 000179, loss = 0.001813
grad_step = 000180, loss = 0.001806
grad_step = 000181, loss = 0.001794
grad_step = 000182, loss = 0.001781
grad_step = 000183, loss = 0.001762
grad_step = 000184, loss = 0.001753
grad_step = 000185, loss = 0.001743
grad_step = 000186, loss = 0.001734
grad_step = 000187, loss = 0.001730
grad_step = 000188, loss = 0.001726
grad_step = 000189, loss = 0.001725
grad_step = 000190, loss = 0.001749
grad_step = 000191, loss = 0.001836
grad_step = 000192, loss = 0.002105
grad_step = 000193, loss = 0.002086
grad_step = 000194, loss = 0.001951
grad_step = 000195, loss = 0.001753
grad_step = 000196, loss = 0.001918
grad_step = 000197, loss = 0.001968
grad_step = 000198, loss = 0.001719
grad_step = 000199, loss = 0.001785
grad_step = 000200, loss = 0.002043
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001898
grad_step = 000202, loss = 0.001714
grad_step = 000203, loss = 0.001778
grad_step = 000204, loss = 0.001870
grad_step = 000205, loss = 0.001741
grad_step = 000206, loss = 0.001673
grad_step = 000207, loss = 0.001778
grad_step = 000208, loss = 0.001785
grad_step = 000209, loss = 0.001673
grad_step = 000210, loss = 0.001680
grad_step = 000211, loss = 0.001748
grad_step = 000212, loss = 0.001689
grad_step = 000213, loss = 0.001629
grad_step = 000214, loss = 0.001680
grad_step = 000215, loss = 0.001684
grad_step = 000216, loss = 0.001646
grad_step = 000217, loss = 0.001616
grad_step = 000218, loss = 0.001658
grad_step = 000219, loss = 0.001715
grad_step = 000220, loss = 0.001640
grad_step = 000221, loss = 0.001595
grad_step = 000222, loss = 0.001651
grad_step = 000223, loss = 0.001675
grad_step = 000224, loss = 0.001668
grad_step = 000225, loss = 0.001583
grad_step = 000226, loss = 0.001687
grad_step = 000227, loss = 0.001793
grad_step = 000228, loss = 0.001676
grad_step = 000229, loss = 0.001621
grad_step = 000230, loss = 0.001745
grad_step = 000231, loss = 0.001643
grad_step = 000232, loss = 0.001603
grad_step = 000233, loss = 0.001640
grad_step = 000234, loss = 0.001620
grad_step = 000235, loss = 0.001662
grad_step = 000236, loss = 0.001565
grad_step = 000237, loss = 0.001605
grad_step = 000238, loss = 0.001580
grad_step = 000239, loss = 0.001607
grad_step = 000240, loss = 0.001575
grad_step = 000241, loss = 0.001551
grad_step = 000242, loss = 0.001577
grad_step = 000243, loss = 0.001548
grad_step = 000244, loss = 0.001571
grad_step = 000245, loss = 0.001526
grad_step = 000246, loss = 0.001540
grad_step = 000247, loss = 0.001533
grad_step = 000248, loss = 0.001531
grad_step = 000249, loss = 0.001533
grad_step = 000250, loss = 0.001506
grad_step = 000251, loss = 0.001515
grad_step = 000252, loss = 0.001500
grad_step = 000253, loss = 0.001502
grad_step = 000254, loss = 0.001505
grad_step = 000255, loss = 0.001496
grad_step = 000256, loss = 0.001503
grad_step = 000257, loss = 0.001493
grad_step = 000258, loss = 0.001492
grad_step = 000259, loss = 0.001496
grad_step = 000260, loss = 0.001484
grad_step = 000261, loss = 0.001487
grad_step = 000262, loss = 0.001479
grad_step = 000263, loss = 0.001472
grad_step = 000264, loss = 0.001469
grad_step = 000265, loss = 0.001459
grad_step = 000266, loss = 0.001454
grad_step = 000267, loss = 0.001448
grad_step = 000268, loss = 0.001439
grad_step = 000269, loss = 0.001437
grad_step = 000270, loss = 0.001433
grad_step = 000271, loss = 0.001432
grad_step = 000272, loss = 0.001442
grad_step = 000273, loss = 0.001469
grad_step = 000274, loss = 0.001519
grad_step = 000275, loss = 0.001631
grad_step = 000276, loss = 0.001561
grad_step = 000277, loss = 0.001506
grad_step = 000278, loss = 0.001414
grad_step = 000279, loss = 0.001403
grad_step = 000280, loss = 0.001452
grad_step = 000281, loss = 0.001457
grad_step = 000282, loss = 0.001431
grad_step = 000283, loss = 0.001381
grad_step = 000284, loss = 0.001386
grad_step = 000285, loss = 0.001427
grad_step = 000286, loss = 0.001426
grad_step = 000287, loss = 0.001411
grad_step = 000288, loss = 0.001377
grad_step = 000289, loss = 0.001355
grad_step = 000290, loss = 0.001355
grad_step = 000291, loss = 0.001366
grad_step = 000292, loss = 0.001382
grad_step = 000293, loss = 0.001391
grad_step = 000294, loss = 0.001398
grad_step = 000295, loss = 0.001379
grad_step = 000296, loss = 0.001365
grad_step = 000297, loss = 0.001344
grad_step = 000298, loss = 0.001330
grad_step = 000299, loss = 0.001324
grad_step = 000300, loss = 0.001325
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001331
grad_step = 000302, loss = 0.001336
grad_step = 000303, loss = 0.001344
grad_step = 000304, loss = 0.001344
grad_step = 000305, loss = 0.001347
grad_step = 000306, loss = 0.001343
grad_step = 000307, loss = 0.001342
grad_step = 000308, loss = 0.001334
grad_step = 000309, loss = 0.001325
grad_step = 000310, loss = 0.001315
grad_step = 000311, loss = 0.001307
grad_step = 000312, loss = 0.001301
grad_step = 000313, loss = 0.001296
grad_step = 000314, loss = 0.001293
grad_step = 000315, loss = 0.001291
grad_step = 000316, loss = 0.001291
grad_step = 000317, loss = 0.001290
grad_step = 000318, loss = 0.001290
grad_step = 000319, loss = 0.001292
grad_step = 000320, loss = 0.001300
grad_step = 000321, loss = 0.001319
grad_step = 000322, loss = 0.001371
grad_step = 000323, loss = 0.001440
grad_step = 000324, loss = 0.001571
grad_step = 000325, loss = 0.001530
grad_step = 000326, loss = 0.001471
grad_step = 000327, loss = 0.001318
grad_step = 000328, loss = 0.001281
grad_step = 000329, loss = 0.001353
grad_step = 000330, loss = 0.001395
grad_step = 000331, loss = 0.001378
grad_step = 000332, loss = 0.001296
grad_step = 000333, loss = 0.001278
grad_step = 000334, loss = 0.001316
grad_step = 000335, loss = 0.001337
grad_step = 000336, loss = 0.001318
grad_step = 000337, loss = 0.001270
grad_step = 000338, loss = 0.001268
grad_step = 000339, loss = 0.001302
grad_step = 000340, loss = 0.001301
grad_step = 000341, loss = 0.001279
grad_step = 000342, loss = 0.001262
grad_step = 000343, loss = 0.001258
grad_step = 000344, loss = 0.001263
grad_step = 000345, loss = 0.001274
grad_step = 000346, loss = 0.001286
grad_step = 000347, loss = 0.001273
grad_step = 000348, loss = 0.001257
grad_step = 000349, loss = 0.001248
grad_step = 000350, loss = 0.001246
grad_step = 000351, loss = 0.001244
grad_step = 000352, loss = 0.001240
grad_step = 000353, loss = 0.001241
grad_step = 000354, loss = 0.001247
grad_step = 000355, loss = 0.001249
grad_step = 000356, loss = 0.001245
grad_step = 000357, loss = 0.001241
grad_step = 000358, loss = 0.001240
grad_step = 000359, loss = 0.001241
grad_step = 000360, loss = 0.001235
grad_step = 000361, loss = 0.001230
grad_step = 000362, loss = 0.001228
grad_step = 000363, loss = 0.001230
grad_step = 000364, loss = 0.001231
grad_step = 000365, loss = 0.001233
grad_step = 000366, loss = 0.001238
grad_step = 000367, loss = 0.001254
grad_step = 000368, loss = 0.001276
grad_step = 000369, loss = 0.001324
grad_step = 000370, loss = 0.001350
grad_step = 000371, loss = 0.001397
grad_step = 000372, loss = 0.001350
grad_step = 000373, loss = 0.001301
grad_step = 000374, loss = 0.001236
grad_step = 000375, loss = 0.001206
grad_step = 000376, loss = 0.001220
grad_step = 000377, loss = 0.001252
grad_step = 000378, loss = 0.001280
grad_step = 000379, loss = 0.001271
grad_step = 000380, loss = 0.001245
grad_step = 000381, loss = 0.001211
grad_step = 000382, loss = 0.001199
grad_step = 000383, loss = 0.001203
grad_step = 000384, loss = 0.001214
grad_step = 000385, loss = 0.001228
grad_step = 000386, loss = 0.001229
grad_step = 000387, loss = 0.001224
grad_step = 000388, loss = 0.001208
grad_step = 000389, loss = 0.001197
grad_step = 000390, loss = 0.001191
grad_step = 000391, loss = 0.001187
grad_step = 000392, loss = 0.001182
grad_step = 000393, loss = 0.001181
grad_step = 000394, loss = 0.001186
grad_step = 000395, loss = 0.001191
grad_step = 000396, loss = 0.001194
grad_step = 000397, loss = 0.001196
grad_step = 000398, loss = 0.001201
grad_step = 000399, loss = 0.001205
grad_step = 000400, loss = 0.001213
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001214
grad_step = 000402, loss = 0.001224
grad_step = 000403, loss = 0.001229
grad_step = 000404, loss = 0.001243
grad_step = 000405, loss = 0.001236
grad_step = 000406, loss = 0.001220
grad_step = 000407, loss = 0.001195
grad_step = 000408, loss = 0.001177
grad_step = 000409, loss = 0.001162
grad_step = 000410, loss = 0.001159
grad_step = 000411, loss = 0.001168
grad_step = 000412, loss = 0.001181
grad_step = 000413, loss = 0.001195
grad_step = 000414, loss = 0.001201
grad_step = 000415, loss = 0.001213
grad_step = 000416, loss = 0.001218
grad_step = 000417, loss = 0.001219
grad_step = 000418, loss = 0.001202
grad_step = 000419, loss = 0.001187
grad_step = 000420, loss = 0.001168
grad_step = 000421, loss = 0.001153
grad_step = 000422, loss = 0.001142
grad_step = 000423, loss = 0.001141
grad_step = 000424, loss = 0.001149
grad_step = 000425, loss = 0.001158
grad_step = 000426, loss = 0.001168
grad_step = 000427, loss = 0.001177
grad_step = 000428, loss = 0.001193
grad_step = 000429, loss = 0.001201
grad_step = 000430, loss = 0.001206
grad_step = 000431, loss = 0.001196
grad_step = 000432, loss = 0.001186
grad_step = 000433, loss = 0.001163
grad_step = 000434, loss = 0.001142
grad_step = 000435, loss = 0.001126
grad_step = 000436, loss = 0.001124
grad_step = 000437, loss = 0.001131
grad_step = 000438, loss = 0.001139
grad_step = 000439, loss = 0.001148
grad_step = 000440, loss = 0.001156
grad_step = 000441, loss = 0.001169
grad_step = 000442, loss = 0.001175
grad_step = 000443, loss = 0.001173
grad_step = 000444, loss = 0.001162
grad_step = 000445, loss = 0.001149
grad_step = 000446, loss = 0.001133
grad_step = 000447, loss = 0.001119
grad_step = 000448, loss = 0.001108
grad_step = 000449, loss = 0.001107
grad_step = 000450, loss = 0.001114
grad_step = 000451, loss = 0.001121
grad_step = 000452, loss = 0.001128
grad_step = 000453, loss = 0.001134
grad_step = 000454, loss = 0.001145
grad_step = 000455, loss = 0.001155
grad_step = 000456, loss = 0.001161
grad_step = 000457, loss = 0.001159
grad_step = 000458, loss = 0.001159
grad_step = 000459, loss = 0.001146
grad_step = 000460, loss = 0.001133
grad_step = 000461, loss = 0.001114
grad_step = 000462, loss = 0.001098
grad_step = 000463, loss = 0.001089
grad_step = 000464, loss = 0.001089
grad_step = 000465, loss = 0.001092
grad_step = 000466, loss = 0.001098
grad_step = 000467, loss = 0.001107
grad_step = 000468, loss = 0.001123
grad_step = 000469, loss = 0.001148
grad_step = 000470, loss = 0.001172
grad_step = 000471, loss = 0.001200
grad_step = 000472, loss = 0.001212
grad_step = 000473, loss = 0.001219
grad_step = 000474, loss = 0.001184
grad_step = 000475, loss = 0.001143
grad_step = 000476, loss = 0.001095
grad_step = 000477, loss = 0.001074
grad_step = 000478, loss = 0.001085
grad_step = 000479, loss = 0.001110
grad_step = 000480, loss = 0.001129
grad_step = 000481, loss = 0.001131
grad_step = 000482, loss = 0.001132
grad_step = 000483, loss = 0.001134
grad_step = 000484, loss = 0.001129
grad_step = 000485, loss = 0.001110
grad_step = 000486, loss = 0.001084
grad_step = 000487, loss = 0.001068
grad_step = 000488, loss = 0.001066
grad_step = 000489, loss = 0.001066
grad_step = 000490, loss = 0.001062
grad_step = 000491, loss = 0.001061
grad_step = 000492, loss = 0.001067
grad_step = 000493, loss = 0.001074
grad_step = 000494, loss = 0.001073
grad_step = 000495, loss = 0.001070
grad_step = 000496, loss = 0.001068
grad_step = 000497, loss = 0.001073
grad_step = 000498, loss = 0.001079
grad_step = 000499, loss = 0.001083
grad_step = 000500, loss = 0.001087
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001092
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

  date_run                              2020-05-11 03:18:32.857265
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.203999
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 03:18:32.862950
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.102496
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 03:18:32.870322
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.128105
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 03:18:32.875429
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.557467
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb071b781d0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352208.6250
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 260875.8750
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 151850.0312
Epoch 4/10

1/1 [==============================] - 0s 95ms/step - loss: 86586.4688
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 49842.9141
Epoch 6/10

1/1 [==============================] - 0s 90ms/step - loss: 30197.8301
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 19516.5859
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 13469.8604
Epoch 9/10

1/1 [==============================] - 0s 95ms/step - loss: 9870.8926
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 7633.1499

  #### Inference Need return ypred, ytrue ######################### 
[[-9.2475966e-02  6.9337950e+00  7.9478426e+00  7.0736365e+00
   6.5071650e+00  8.9328299e+00  6.5230875e+00  6.0044608e+00
   8.6862020e+00  6.4202867e+00  5.8719726e+00  7.9486065e+00
   7.3142505e+00  5.6990790e+00  7.4221640e+00  5.4902678e+00
   7.4595809e+00  8.5773220e+00  8.7259903e+00  7.1578250e+00
   7.7428837e+00  7.3288517e+00  8.0095425e+00  7.3853970e+00
   8.1781988e+00  6.9453645e+00  5.9798384e+00  6.8261547e+00
   6.1058974e+00  6.5820560e+00  7.9487238e+00  9.2624912e+00
   7.3118310e+00  6.2835655e+00  6.7867656e+00  8.6066637e+00
   6.7439818e+00  6.2989898e+00  8.0325384e+00  4.9504919e+00
   8.5164633e+00  7.6030655e+00  8.3762121e+00  6.1075478e+00
   7.1215978e+00  8.4616814e+00  6.1757622e+00  7.1566896e+00
   6.5471377e+00  8.0318661e+00  7.9939246e+00  8.2714806e+00
   6.9575505e+00  7.8671608e+00  8.8890533e+00  7.6790237e+00
   7.4713020e+00  8.2183952e+00  6.6723557e+00  8.0749159e+00
   1.5148300e+00 -1.8357873e-01  6.3731968e-01  9.9803758e-01
  -8.1376910e-01  6.9046170e-01 -4.7721997e-01 -5.1148140e-01
   2.5098017e-01 -3.2098591e-03 -3.7011820e-01  4.7496796e-01
   4.3000689e-01 -1.0660863e+00 -7.2264338e-01 -1.6444185e+00
   4.1560233e-03 -3.6080843e-01 -5.3058463e-01  3.3016530e-01
  -5.4330623e-01 -2.6461697e-01  4.5589566e-01  1.2339722e+00
   1.1857049e+00 -3.8219273e-02  1.2346170e+00 -7.1119118e-01
   5.1029289e-01 -1.0078938e+00 -4.5620561e-02  3.5647374e-01
   2.2163031e-01 -1.3180059e+00  5.7159108e-01  7.0883471e-01
  -6.3669717e-01 -1.7158041e+00  1.1348369e+00 -1.5108186e+00
  -1.4327015e+00  1.3676637e-01  5.7997137e-01 -9.5880949e-01
  -2.3086526e+00  5.0171721e-01 -9.2342645e-01 -6.3287640e-01
   1.5924917e+00  1.1677524e+00  1.8348879e-01 -4.3096000e-01
  -8.7665117e-01  3.1905609e-01  9.9507880e-01  1.4464748e-01
   9.7257304e-01  1.5466914e+00  2.0682663e-02 -1.6864688e+00
   3.6428332e-01  1.2731338e+00  1.6802840e-01  8.9302701e-01
  -2.5130314e-01  6.5664434e-01 -6.4704120e-01 -4.5788628e-01
  -1.9551551e+00  1.1126764e+00  7.5351560e-01 -3.2306999e-02
  -4.9812642e-01 -1.3233256e+00 -1.0404516e+00  1.4669785e-01
   3.2234424e-01 -1.3274411e+00  3.9678729e-01  1.8034075e-01
   6.9135433e-01  9.1195345e-02 -8.1549501e-01 -2.4369699e-01
  -6.8325326e-02 -5.1117837e-03  4.7417733e-01 -7.9255772e-01
   6.3528717e-01  3.2727736e-01  1.1325514e+00 -1.1556146e+00
  -5.1974714e-02  1.0498534e+00 -2.3192999e-01 -1.4791241e+00
  -4.0199104e-01  1.8394424e-01 -8.7768996e-01 -9.7703958e-01
   4.8196223e-01 -5.8790815e-01  4.1148967e-01  2.3982865e-01
  -5.6793606e-01  9.5720124e-01  9.8685741e-02  1.2961932e+00
  -7.9413545e-01 -4.8278314e-01  8.2977068e-01 -1.0533426e+00
  -1.5243801e+00 -4.3496877e-02  1.5754607e-01 -2.4397594e-01
   1.9705769e-01  4.4885164e-01  9.1033751e-01 -1.2390332e-01
   1.0883546e-01  7.3294191e+00  8.8714523e+00  7.7575321e+00
   7.2419310e+00  7.9975548e+00  7.9192095e+00  8.2632380e+00
   7.1580944e+00  7.4685864e+00  6.3025365e+00  8.1330471e+00
   8.2630539e+00  6.0398045e+00  7.3138556e+00  7.3157721e+00
   7.8811731e+00  7.8018012e+00  7.5287437e+00  6.4739056e+00
   7.8784966e+00  8.2667313e+00  6.4966693e+00  8.3411474e+00
   7.7837076e+00  8.6699467e+00  7.3331361e+00  6.8190765e+00
   7.3536501e+00  8.2482939e+00  7.9976244e+00  8.3425407e+00
   8.3023376e+00  8.1966991e+00  8.7071733e+00  7.7143159e+00
   8.4646196e+00  7.1610980e+00  7.1026530e+00  8.0249357e+00
   8.4886456e+00  7.3608952e+00  5.8003035e+00  7.9594188e+00
   7.5613923e+00  7.3999701e+00  8.1712484e+00  6.9880471e+00
   7.8511004e+00  8.0803709e+00  7.7207699e+00  7.4310060e+00
   7.1483188e+00  8.9110565e+00  7.3111548e+00  9.2656183e+00
   7.4430628e+00  8.0954447e+00  8.1153641e+00  7.6876431e+00
   3.0324626e-01  1.8831450e-01  1.2714456e+00  2.2087610e-01
   6.6982019e-01  3.4497154e-01  1.4975138e+00  2.6332297e+00
   3.3346236e-01  1.9758183e-01  1.2578388e+00  1.4210045e+00
   3.0360365e-01  2.8766739e-01  6.7472148e-01  1.5526789e-01
   1.1423583e+00  3.5547811e-01  4.0588456e-01  2.8166194e+00
   1.5320802e-01  2.5832047e+00  2.2997460e+00  9.6440178e-01
   1.8345716e+00  2.4701977e-01  2.6605563e+00  1.9821618e+00
   1.9899991e+00  2.0977488e+00  9.2840648e-01  7.7262527e-01
   1.5513399e+00  5.0756413e-01  1.6186266e+00  2.7867895e-01
   2.0995939e-01  4.1675234e-01  1.3832814e+00  3.2908738e-01
   2.1291537e+00  8.4776849e-01  1.5702734e+00  5.4667729e-01
   3.1686335e+00  1.8587592e+00  1.4331285e+00  9.1100109e-01
   6.5165710e-01  1.2749288e+00  1.7940868e+00  5.4769129e-01
   4.3031788e-01  2.9614335e-01  2.9576474e-01  2.6442363e+00
   5.2316940e-01  2.0201116e+00  1.3047585e+00  3.4526497e-01
   2.1698895e+00  1.2322614e+00  9.8418510e-01  2.8607392e-01
   1.4854472e+00  5.2550423e-01  1.1437821e+00  2.7303424e+00
   6.7031854e-01  1.0612395e+00  1.7403481e+00  1.0266664e+00
   2.7577004e+00  1.4376599e-01  1.3522363e-01  1.3489521e+00
   2.9536396e-01  1.2509518e+00  4.3246537e-01  4.4136739e-01
   1.2248924e+00  5.3759891e-01  1.1713940e+00  1.1050658e+00
   2.1603627e+00  2.8799863e+00  1.7966435e+00  7.7970034e-01
   7.3985231e-01  1.3854313e-01  3.0950031e+00  1.8882754e+00
   6.0979307e-01  1.4413314e+00  1.6465905e+00  3.2505846e-01
   1.2097003e+00  1.2751011e+00  3.9654982e-01  3.3699983e-01
   9.3825603e-01  1.7321289e-01  1.7353234e+00  9.7315741e-01
   3.8207263e-01  1.1513094e+00  1.7490878e+00  7.5853658e-01
   3.3353662e-01  2.8827748e+00  1.4077352e+00  2.6614079e+00
   5.7917506e-01  2.3393688e+00  1.9930836e+00  2.1283293e+00
   9.6011662e-01  1.2212400e+00  1.9151076e+00  5.4409230e-01
   1.2389092e+01 -8.6879559e+00 -8.0706692e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:18:41.524232
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    95.077
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 03:18:41.527956
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9059.19
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 03:18:41.531091
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.0516
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 03:18:41.534137
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -810.314
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fafe3c543c8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 03:18:56.804368
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 03:18:56.807645
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 03:18:56.810702
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 03:18:56.813761
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
0   2020-05-11 03:18:32.857265  ...    mean_absolute_error
1   2020-05-11 03:18:32.862950  ...     mean_squared_error
2   2020-05-11 03:18:32.870322  ...  median_absolute_error
3   2020-05-11 03:18:32.875429  ...               r2_score
4   2020-05-11 03:18:41.524232  ...    mean_absolute_error
5   2020-05-11 03:18:41.527956  ...     mean_squared_error
6   2020-05-11 03:18:41.531091  ...  median_absolute_error
7   2020-05-11 03:18:41.534137  ...               r2_score
8   2020-05-11 03:18:56.804368  ...    mean_absolute_error
9   2020-05-11 03:18:56.807645  ...     mean_squared_error
10  2020-05-11 03:18:56.810702  ...  median_absolute_error
11  2020-05-11 03:18:56.813761  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 118, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

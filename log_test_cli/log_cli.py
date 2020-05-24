
  test_cli /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_cli', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_cli 

  # Testing Command Line System   





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

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

model_keras.keras_gan
model_keras.preprocess
model_keras.nbeats
model_keras.01_deepctr
model_keras.textvae
model_keras.namentity_crm_bilstm_dataloader
model_keras.Autokeras
model_keras.charcnn_zhang
model_keras.charcnn
model_keras.namentity_crm_bilstm
model_keras.textcnn
model_keras.armdn
model_keras.02_cnn
model_tf.1_lstm
model_tf.temporal_fusion_google
model_gluon.gluon_automl
model_gluon.fb_prophet
model_gluon.gluonts_model
model_sklearn.model_sklearn
model_sklearn.model_lightgbm
model_tch.nbeats
model_tch.transformer_classifier
model_tch.matchzoo_models
model_tch.torchhub
model_tch.03_nbeats_dataloader
model_tch.transformer_sentence
model_tchtorch_vae
model_tch.pplm
model_tch.textcnn
model_tch.mlp





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
fit

  ##### Load JSON /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.json 

  ##### Init model_tf.1_lstm {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'} 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  ##### Fit <mlmodels.model_tf.1_lstm.Model object at 0x7fa28e020d30> 
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

  ##### Save <tensorflow.python.client.session.Session object at 0x7fa25b53be10> 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 





 ************************************************************************************************************************
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
predict

  ##### Load JSON /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.json test 

  ##### Init model_tf.1_lstm {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'} 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  ##### Load from disk: {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'} 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/model
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/model

  ##### Predict: <tensorflow.python.client.session.Session object at 0x7f44a8680668> 
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f98e0923198> 

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
 [-0.02438341  0.10807145  0.10550625  0.15814915  0.02857511 -0.02515273]
 [ 0.13108166  0.2682355   0.48506379  0.08445184  0.18575586  0.04095699]
 [-0.05666401  0.01467365  0.21227154  0.13719177  0.1670036  -0.12421441]
 [ 0.26899418  0.07489403  0.05712566  0.07869834 -0.0236113   0.12078637]
 [ 0.80030125  0.12234835  0.80944383  0.04091521 -0.54064709  0.08307657]
 [-0.34984612  0.02710668  0.05613611  0.59028268  0.37995711  0.25958377]
 [ 0.16510792 -0.4058792   0.00950995  1.00683653  1.06178951 -0.28460333]
 [ 0.08663408  0.47609627  0.68091047  0.21048802 -0.1591938   0.02970184]
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
{'loss': 0.43234976939857006, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model
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
{'loss': 0.40971432998776436, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model





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

  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f02f7dc3198> 

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
 [ 0.06282453 -0.05313327  0.12346976  0.19581944  0.03217344 -0.01324819]
 [ 0.0160218   0.18270451  0.04203121 -0.05025528  0.05763914 -0.12187491]
 [ 0.12656273 -0.09441148 -0.07166872 -0.2052886   0.11298308  0.2448609 ]
 [ 0.25585005 -0.35802901 -0.17953666  0.30815727 -0.05948658  0.22078529]
 [ 0.23589936  0.08879966 -0.03533889 -0.20129386  0.25604892 -0.55656308]
 [ 0.50202042  0.01325113  0.42097899 -0.043059    0.45790341 -0.02120502]
 [ 0.26632956 -0.04323467 -0.51009506  0.28134936  0.46827382 -0.24652599]
 [ 0.21255574 -0.13649595 -0.28754756 -0.07484841  0.14590538 -0.00979418]
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
{'loss': 0.568153090775013, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model
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
{'loss': 0.44203492626547813, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model





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
[32m[I 2020-05-24 23:19:06,816][0m Finished trial#0 resulted in value: 0.3094245418906212. Current best value is 0.3094245418906212 with parameters: {'learning_rate': 0.002063323763262461, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-24 23:19:08,948][0m Finished trial#1 resulted in value: 12.162828803062439. Current best value is 0.3094245418906212 with parameters: {'learning_rate': 0.002063323763262461, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/optim_1lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 

  #############  OPTIMIZATION End ############### 

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.002063323763262461, 'num_layers': 3, 'size': 6, 'size_layer': 256, 'output_size': 6, 'timestep': 5, 'epoch': 2, 'best_value': 0.3094245418906212, 'model_name': None} 





 ************************************************************************************************************************
ml_optim --do test   --model_uri model_tf.1_lstm   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-24 23:19:16,680][0m Finished trial#0 resulted in value: 0.3042147383093834. Current best value is 0.3042147383093834 with parameters: {'learning_rate': 0.0014397216467166077, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-24 23:19:18,334][0m Finished trial#1 resulted in value: 0.3320719674229622. Current best value is 0.3042147383093834 with parameters: {'learning_rate': 0.0014397216467166077, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/optim_1lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 





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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f49d364a898> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 23:19:36.938067
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-24 23:19:36.942087
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-24 23:19:36.945648
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-24 23:19:36.948761
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f49d364ac88> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 350905.2812
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 242814.2969
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 142671.0312
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 76615.8594
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 41592.7344
Epoch 6/10

1/1 [==============================] - 0s 90ms/step - loss: 24549.7246
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 15900.6777
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 11047.0020
Epoch 9/10

1/1 [==============================] - 0s 90ms/step - loss: 8187.8677
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 6358.0278

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.0659575e-01  3.3527547e-01 -9.5304191e-01  1.2531939e+00
  -3.7009555e-01  1.0734018e+00  1.1747799e+00  2.3673320e-01
   7.3496664e-01  3.6824822e-02  1.1544045e+00  2.0940952e+00
  -1.2161939e+00 -8.7400097e-01 -4.9305260e-03  1.4452312e+00
  -5.7555622e-01 -1.4688185e-01 -1.4439982e-01 -2.7662995e-01
  -1.2681348e+00 -2.5396630e-01 -1.7146854e+00 -2.1164890e-01
  -1.2196727e+00  2.0180303e-01 -1.3298347e+00 -6.1135113e-02
   1.1212670e+00  7.8650671e-01  2.2172853e-01  1.2330970e+00
   1.4949630e+00 -1.0334352e+00 -1.5680962e+00 -9.9428886e-01
  -5.4187632e-01 -2.1186841e-01 -6.1301136e-01  8.4326386e-01
   1.6124150e+00 -6.0042310e-01  2.3564401e-01 -5.3248930e-01
  -8.5194910e-01  4.5283407e-01 -1.3723628e-01 -7.0651102e-01
   1.2654846e+00  8.8404375e-01  2.2046027e+00  3.0313396e-01
   1.0752811e+00  1.7020073e+00  1.8763297e+00 -2.0347822e+00
  -3.3686221e-01 -4.1245636e-01  6.0665846e-02 -2.2012162e-01
  -1.2667544e+00 -2.0976717e+00  1.3846123e+00  7.6055735e-01
   2.3888516e+00 -9.9080980e-01  1.6168884e+00  1.4240915e+00
  -6.5803659e-01  1.8941455e+00 -1.4900045e+00 -4.7160429e-01
  -9.4763529e-01 -3.2704434e-01  9.2732006e-01 -2.1673217e+00
  -3.9841267e-01  1.1230886e+00 -7.2181433e-01 -5.8211809e-01
   5.4090893e-01 -2.1801519e+00  2.0871395e-01 -5.7659283e-02
  -4.7115231e-01 -1.8123291e+00 -1.6598281e-01 -1.6441574e+00
  -2.1917585e-01 -9.6684998e-01  2.1794063e-01 -1.2370522e+00
  -4.7505009e-01  1.3955789e+00  2.7141923e-01 -5.7445467e-01
   1.1478420e+00  2.7791825e-01 -5.9360409e-01 -1.2129815e+00
  -1.9270815e-02  3.1736892e-01  2.1692226e+00  2.7786922e-01
   5.6729174e-01  8.3190721e-01 -5.2450645e-01 -6.8854445e-01
   1.6544789e-02 -5.2560353e-01 -9.2880368e-01  1.4109427e-01
  -7.7869868e-01 -1.4182220e+00 -7.1048510e-01  1.3432578e+00
  -6.8505064e-02 -4.1945744e-01  3.0673516e-01 -1.5047983e+00
  -1.9582416e-01  8.4439478e+00  9.6830826e+00  9.0289164e+00
   8.5054493e+00  9.0981283e+00  9.0783119e+00  9.4197187e+00
   8.4165878e+00  7.2250791e+00  7.9259238e+00  9.3696289e+00
   7.6106896e+00  7.5577598e+00  7.5061660e+00  7.3901706e+00
   7.8129578e+00  6.6470737e+00  6.6655035e+00  9.3026037e+00
   7.3658633e+00  7.3828325e+00  7.9336371e+00  8.3471632e+00
   8.2875433e+00  6.3662157e+00  9.2626066e+00  8.6224937e+00
   8.3371325e+00  7.7085443e+00  8.1894751e+00  7.6541190e+00
   8.4065952e+00  7.3133492e+00  8.3001118e+00  7.6428704e+00
   8.4847374e+00  1.0044748e+01  9.7341280e+00  6.8893309e+00
   9.3615398e+00  7.5579505e+00  6.6725683e+00  6.1860552e+00
   8.1716852e+00  8.0730038e+00  8.0266104e+00  8.0106945e+00
   8.5068169e+00  7.8492746e+00  8.7137775e+00  9.6861095e+00
   8.4477959e+00  6.9624023e+00  9.4495621e+00  9.3711252e+00
   8.3151264e+00  6.4905539e+00  7.7063046e+00  7.6036658e+00
   1.7537482e+00  7.2853124e-01  2.8930035e+00  4.3708479e-01
   6.2600428e-01  1.4763880e+00  1.2418031e+00  2.6341872e+00
   8.6595058e-01  9.5649731e-01  2.6100221e+00  5.8314270e-01
   1.4510193e+00  1.0317441e+00  3.9148426e-01  1.0956908e+00
   2.1174855e+00  2.8468573e-01  2.2677424e+00  1.3812096e+00
   3.7261593e-01  2.4215573e-01  6.3603216e-01  3.9511621e-01
   1.6622660e+00  3.4318626e-01  1.2909908e+00  1.8389313e+00
   2.5943339e-01  1.2912003e+00  1.7532690e+00  8.9391637e-01
   8.2327759e-01  6.4759374e-01  6.0015243e-01  7.3797470e-01
   2.2886848e-01  6.8581975e-01  3.6793220e-01  1.7934344e+00
   7.9344517e-01  1.1384740e+00  1.3992763e+00  5.3345460e-01
   1.2790959e+00  1.6106838e+00  1.6665905e+00  1.0066954e+00
   5.6458086e-01  1.4794617e+00  7.1134198e-01  2.0008531e+00
   5.4308826e-01  3.6054003e-01  2.9820949e-01  8.2137299e-01
   3.6787796e-01  1.5489634e+00  2.5779867e-01  8.5195553e-01
   2.7595830e-01  2.3408175e+00  1.1158632e+00  7.7467817e-01
   1.8059363e+00  1.6240299e+00  1.3242581e+00  3.4179196e+00
   2.6296191e+00  1.7585130e+00  2.3313084e+00  1.9107461e-01
   1.4687567e+00  3.6904776e-01  1.8489815e+00  1.0182099e+00
   1.8454580e+00  5.8759105e-01  1.8049235e+00  1.2300789e+00
   1.4928880e+00  6.8518257e-01  2.0454597e-01  1.5759985e+00
   5.4079014e-01  1.5800083e-01  1.3885927e-01  5.3280598e-01
   5.5665684e-01  1.4611661e-01  1.9479384e+00  1.7675501e+00
   2.5666018e+00  2.2874284e+00  2.7887621e+00  1.7524490e+00
   3.6114306e+00  1.0046499e+00  2.4389586e+00  1.6609437e+00
   3.6883038e-01  1.6830170e-01  2.4109054e-01  5.6757927e-01
   1.5049242e+00  8.8249332e-01  3.0959759e+00  5.8880705e-01
   2.2039814e+00  4.8560905e-01  1.4331908e+00  4.0743279e-01
   5.4833657e-01  2.4474955e+00  8.1167126e-01  9.2549044e-01
   5.0167549e-01  4.1507417e-01  3.4536248e-01  2.2362096e+00
   1.3824904e-01  1.0182451e+01  8.0018806e+00  9.3336334e+00
   9.2745342e+00  8.5422897e+00  8.2020874e+00  8.8560181e+00
   9.0466528e+00  8.8333559e+00  7.1987562e+00  7.7095294e+00
   7.0103674e+00  8.4181976e+00  9.3399658e+00  8.6235352e+00
   6.5776238e+00  9.1768265e+00  8.3906984e+00  8.6415234e+00
   8.9415855e+00  8.8565283e+00  8.7061892e+00  6.9859638e+00
   9.0943785e+00  9.1001091e+00  9.2403374e+00  9.2090454e+00
   6.7927785e+00  9.6097507e+00  1.0144173e+01  9.3114157e+00
   8.9483471e+00  7.3784184e+00  7.9935675e+00  9.2751923e+00
   7.4261661e+00  1.0579910e+01  7.9121532e+00  9.7450562e+00
   9.7948389e+00  6.7685871e+00  7.1993537e+00  7.6519074e+00
   8.3873606e+00  7.5812840e+00  8.8572874e+00  1.0147879e+01
   8.2293692e+00  7.4643397e+00  8.3612118e+00  8.3784151e+00
   8.9131374e+00  8.3389864e+00  9.1715841e+00  8.5548487e+00
   7.4475102e+00  9.1810284e+00  1.0194931e+01  8.6204634e+00
  -1.4095037e+01 -8.0067787e+00  8.1038208e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 23:19:46.042007
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   93.1807
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-24 23:19:46.046272
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8704.95
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-24 23:19:46.049460
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   93.7988
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-24 23:19:46.053424
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   -778.59
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139954438806496
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139953480109136
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139953480109640
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139953480220800
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139953480221304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139953480221808

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f49c0e45390> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.631857
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.592681
grad_step = 000002, loss = 0.561792
grad_step = 000003, loss = 0.523767
grad_step = 000004, loss = 0.479162
grad_step = 000005, loss = 0.432998
grad_step = 000006, loss = 0.401167
grad_step = 000007, loss = 0.398379
grad_step = 000008, loss = 0.378751
grad_step = 000009, loss = 0.345496
grad_step = 000010, loss = 0.317214
grad_step = 000011, loss = 0.297799
grad_step = 000012, loss = 0.284414
grad_step = 000013, loss = 0.273414
grad_step = 000014, loss = 0.262090
grad_step = 000015, loss = 0.249801
grad_step = 000016, loss = 0.237350
grad_step = 000017, loss = 0.226551
grad_step = 000018, loss = 0.218304
grad_step = 000019, loss = 0.210257
grad_step = 000020, loss = 0.200637
grad_step = 000021, loss = 0.190751
grad_step = 000022, loss = 0.181844
grad_step = 000023, loss = 0.173796
grad_step = 000024, loss = 0.166253
grad_step = 000025, loss = 0.159030
grad_step = 000026, loss = 0.151645
grad_step = 000027, loss = 0.144006
grad_step = 000028, loss = 0.136678
grad_step = 000029, loss = 0.130121
grad_step = 000030, loss = 0.124249
grad_step = 000031, loss = 0.118691
grad_step = 000032, loss = 0.113116
grad_step = 000033, loss = 0.107502
grad_step = 000034, loss = 0.102084
grad_step = 000035, loss = 0.097016
grad_step = 000036, loss = 0.092215
grad_step = 000037, loss = 0.087561
grad_step = 000038, loss = 0.082995
grad_step = 000039, loss = 0.078596
grad_step = 000040, loss = 0.074514
grad_step = 000041, loss = 0.070767
grad_step = 000042, loss = 0.067206
grad_step = 000043, loss = 0.063667
grad_step = 000044, loss = 0.060194
grad_step = 000045, loss = 0.056945
grad_step = 000046, loss = 0.053946
grad_step = 000047, loss = 0.051091
grad_step = 000048, loss = 0.048306
grad_step = 000049, loss = 0.045642
grad_step = 000050, loss = 0.043199
grad_step = 000051, loss = 0.040957
grad_step = 000052, loss = 0.038796
grad_step = 000053, loss = 0.036654
grad_step = 000054, loss = 0.034599
grad_step = 000055, loss = 0.032700
grad_step = 000056, loss = 0.030938
grad_step = 000057, loss = 0.029260
grad_step = 000058, loss = 0.027649
grad_step = 000059, loss = 0.026139
grad_step = 000060, loss = 0.024737
grad_step = 000061, loss = 0.023418
grad_step = 000062, loss = 0.022153
grad_step = 000063, loss = 0.020951
grad_step = 000064, loss = 0.019827
grad_step = 000065, loss = 0.018776
grad_step = 000066, loss = 0.017779
grad_step = 000067, loss = 0.016832
grad_step = 000068, loss = 0.015941
grad_step = 000069, loss = 0.015111
grad_step = 000070, loss = 0.014327
grad_step = 000071, loss = 0.013579
grad_step = 000072, loss = 0.012874
grad_step = 000073, loss = 0.012218
grad_step = 000074, loss = 0.011603
grad_step = 000075, loss = 0.011016
grad_step = 000076, loss = 0.010459
grad_step = 000077, loss = 0.009938
grad_step = 000078, loss = 0.009450
grad_step = 000079, loss = 0.008984
grad_step = 000080, loss = 0.008542
grad_step = 000081, loss = 0.008128
grad_step = 000082, loss = 0.007738
grad_step = 000083, loss = 0.007370
grad_step = 000084, loss = 0.007023
grad_step = 000085, loss = 0.006697
grad_step = 000086, loss = 0.006391
grad_step = 000087, loss = 0.006100
grad_step = 000088, loss = 0.005824
grad_step = 000089, loss = 0.005567
grad_step = 000090, loss = 0.005326
grad_step = 000091, loss = 0.005097
grad_step = 000092, loss = 0.004882
grad_step = 000093, loss = 0.004681
grad_step = 000094, loss = 0.004494
grad_step = 000095, loss = 0.004316
grad_step = 000096, loss = 0.004151
grad_step = 000097, loss = 0.003996
grad_step = 000098, loss = 0.003851
grad_step = 000099, loss = 0.003715
grad_step = 000100, loss = 0.003588
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003470
grad_step = 000102, loss = 0.003360
grad_step = 000103, loss = 0.003258
grad_step = 000104, loss = 0.003163
grad_step = 000105, loss = 0.003074
grad_step = 000106, loss = 0.002992
grad_step = 000107, loss = 0.002916
grad_step = 000108, loss = 0.002846
grad_step = 000109, loss = 0.002781
grad_step = 000110, loss = 0.002720
grad_step = 000111, loss = 0.002664
grad_step = 000112, loss = 0.002613
grad_step = 000113, loss = 0.002566
grad_step = 000114, loss = 0.002522
grad_step = 000115, loss = 0.002482
grad_step = 000116, loss = 0.002445
grad_step = 000117, loss = 0.002411
grad_step = 000118, loss = 0.002380
grad_step = 000119, loss = 0.002352
grad_step = 000120, loss = 0.002325
grad_step = 000121, loss = 0.002301
grad_step = 000122, loss = 0.002280
grad_step = 000123, loss = 0.002260
grad_step = 000124, loss = 0.002241
grad_step = 000125, loss = 0.002225
grad_step = 000126, loss = 0.002209
grad_step = 000127, loss = 0.002195
grad_step = 000128, loss = 0.002183
grad_step = 000129, loss = 0.002171
grad_step = 000130, loss = 0.002160
grad_step = 000131, loss = 0.002150
grad_step = 000132, loss = 0.002141
grad_step = 000133, loss = 0.002133
grad_step = 000134, loss = 0.002125
grad_step = 000135, loss = 0.002118
grad_step = 000136, loss = 0.002112
grad_step = 000137, loss = 0.002106
grad_step = 000138, loss = 0.002100
grad_step = 000139, loss = 0.002095
grad_step = 000140, loss = 0.002090
grad_step = 000141, loss = 0.002085
grad_step = 000142, loss = 0.002081
grad_step = 000143, loss = 0.002077
grad_step = 000144, loss = 0.002073
grad_step = 000145, loss = 0.002070
grad_step = 000146, loss = 0.002066
grad_step = 000147, loss = 0.002063
grad_step = 000148, loss = 0.002060
grad_step = 000149, loss = 0.002057
grad_step = 000150, loss = 0.002054
grad_step = 000151, loss = 0.002051
grad_step = 000152, loss = 0.002048
grad_step = 000153, loss = 0.002045
grad_step = 000154, loss = 0.002042
grad_step = 000155, loss = 0.002040
grad_step = 000156, loss = 0.002037
grad_step = 000157, loss = 0.002034
grad_step = 000158, loss = 0.002031
grad_step = 000159, loss = 0.002029
grad_step = 000160, loss = 0.002026
grad_step = 000161, loss = 0.002023
grad_step = 000162, loss = 0.002020
grad_step = 000163, loss = 0.002017
grad_step = 000164, loss = 0.002014
grad_step = 000165, loss = 0.002011
grad_step = 000166, loss = 0.002008
grad_step = 000167, loss = 0.002005
grad_step = 000168, loss = 0.002001
grad_step = 000169, loss = 0.001997
grad_step = 000170, loss = 0.001994
grad_step = 000171, loss = 0.001989
grad_step = 000172, loss = 0.001985
grad_step = 000173, loss = 0.001980
grad_step = 000174, loss = 0.001975
grad_step = 000175, loss = 0.001969
grad_step = 000176, loss = 0.001963
grad_step = 000177, loss = 0.001959
grad_step = 000178, loss = 0.001955
grad_step = 000179, loss = 0.001949
grad_step = 000180, loss = 0.001942
grad_step = 000181, loss = 0.001934
grad_step = 000182, loss = 0.001929
grad_step = 000183, loss = 0.001936
grad_step = 000184, loss = 0.001972
grad_step = 000185, loss = 0.001995
grad_step = 000186, loss = 0.001930
grad_step = 000187, loss = 0.001922
grad_step = 000188, loss = 0.001926
grad_step = 000189, loss = 0.001886
grad_step = 000190, loss = 0.001899
grad_step = 000191, loss = 0.001917
grad_step = 000192, loss = 0.001874
grad_step = 000193, loss = 0.001839
grad_step = 000194, loss = 0.001854
grad_step = 000195, loss = 0.001875
grad_step = 000196, loss = 0.001996
grad_step = 000197, loss = 0.002437
grad_step = 000198, loss = 0.002471
grad_step = 000199, loss = 0.002100
grad_step = 000200, loss = 0.002189
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002092
grad_step = 000202, loss = 0.002275
grad_step = 000203, loss = 0.001895
grad_step = 000204, loss = 0.002161
grad_step = 000205, loss = 0.002086
grad_step = 000206, loss = 0.001930
grad_step = 000207, loss = 0.002130
grad_step = 000208, loss = 0.001936
grad_step = 000209, loss = 0.001958
grad_step = 000210, loss = 0.002015
grad_step = 000211, loss = 0.001836
grad_step = 000212, loss = 0.002008
grad_step = 000213, loss = 0.001841
grad_step = 000214, loss = 0.001925
grad_step = 000215, loss = 0.001885
grad_step = 000216, loss = 0.001847
grad_step = 000217, loss = 0.001906
grad_step = 000218, loss = 0.001819
grad_step = 000219, loss = 0.001877
grad_step = 000220, loss = 0.001835
grad_step = 000221, loss = 0.001829
grad_step = 000222, loss = 0.001841
grad_step = 000223, loss = 0.001798
grad_step = 000224, loss = 0.001832
grad_step = 000225, loss = 0.001799
grad_step = 000226, loss = 0.001803
grad_step = 000227, loss = 0.001810
grad_step = 000228, loss = 0.001779
grad_step = 000229, loss = 0.001809
grad_step = 000230, loss = 0.001778
grad_step = 000231, loss = 0.001793
grad_step = 000232, loss = 0.001785
grad_step = 000233, loss = 0.001781
grad_step = 000234, loss = 0.001785
grad_step = 000235, loss = 0.001774
grad_step = 000236, loss = 0.001784
grad_step = 000237, loss = 0.001769
grad_step = 000238, loss = 0.001779
grad_step = 000239, loss = 0.001772
grad_step = 000240, loss = 0.001769
grad_step = 000241, loss = 0.001775
grad_step = 000242, loss = 0.001765
grad_step = 000243, loss = 0.001770
grad_step = 000244, loss = 0.001765
grad_step = 000245, loss = 0.001765
grad_step = 000246, loss = 0.001764
grad_step = 000247, loss = 0.001761
grad_step = 000248, loss = 0.001764
grad_step = 000249, loss = 0.001758
grad_step = 000250, loss = 0.001760
grad_step = 000251, loss = 0.001759
grad_step = 000252, loss = 0.001756
grad_step = 000253, loss = 0.001757
grad_step = 000254, loss = 0.001754
grad_step = 000255, loss = 0.001755
grad_step = 000256, loss = 0.001753
grad_step = 000257, loss = 0.001751
grad_step = 000258, loss = 0.001752
grad_step = 000259, loss = 0.001750
grad_step = 000260, loss = 0.001749
grad_step = 000261, loss = 0.001748
grad_step = 000262, loss = 0.001747
grad_step = 000263, loss = 0.001747
grad_step = 000264, loss = 0.001745
grad_step = 000265, loss = 0.001745
grad_step = 000266, loss = 0.001744
grad_step = 000267, loss = 0.001743
grad_step = 000268, loss = 0.001742
grad_step = 000269, loss = 0.001741
grad_step = 000270, loss = 0.001740
grad_step = 000271, loss = 0.001739
grad_step = 000272, loss = 0.001738
grad_step = 000273, loss = 0.001737
grad_step = 000274, loss = 0.001736
grad_step = 000275, loss = 0.001735
grad_step = 000276, loss = 0.001735
grad_step = 000277, loss = 0.001734
grad_step = 000278, loss = 0.001733
grad_step = 000279, loss = 0.001732
grad_step = 000280, loss = 0.001731
grad_step = 000281, loss = 0.001730
grad_step = 000282, loss = 0.001729
grad_step = 000283, loss = 0.001728
grad_step = 000284, loss = 0.001728
grad_step = 000285, loss = 0.001727
grad_step = 000286, loss = 0.001726
grad_step = 000287, loss = 0.001725
grad_step = 000288, loss = 0.001724
grad_step = 000289, loss = 0.001723
grad_step = 000290, loss = 0.001723
grad_step = 000291, loss = 0.001722
grad_step = 000292, loss = 0.001721
grad_step = 000293, loss = 0.001720
grad_step = 000294, loss = 0.001720
grad_step = 000295, loss = 0.001719
grad_step = 000296, loss = 0.001720
grad_step = 000297, loss = 0.001721
grad_step = 000298, loss = 0.001724
grad_step = 000299, loss = 0.001728
grad_step = 000300, loss = 0.001740
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001752
grad_step = 000302, loss = 0.001780
grad_step = 000303, loss = 0.001770
grad_step = 000304, loss = 0.001755
grad_step = 000305, loss = 0.001719
grad_step = 000306, loss = 0.001718
grad_step = 000307, loss = 0.001740
grad_step = 000308, loss = 0.001738
grad_step = 000309, loss = 0.001722
grad_step = 000310, loss = 0.001707
grad_step = 000311, loss = 0.001709
grad_step = 000312, loss = 0.001723
grad_step = 000313, loss = 0.001734
grad_step = 000314, loss = 0.001742
grad_step = 000315, loss = 0.001732
grad_step = 000316, loss = 0.001710
grad_step = 000317, loss = 0.001701
grad_step = 000318, loss = 0.001705
grad_step = 000319, loss = 0.001716
grad_step = 000320, loss = 0.001718
grad_step = 000321, loss = 0.001707
grad_step = 000322, loss = 0.001697
grad_step = 000323, loss = 0.001699
grad_step = 000324, loss = 0.001707
grad_step = 000325, loss = 0.001707
grad_step = 000326, loss = 0.001698
grad_step = 000327, loss = 0.001692
grad_step = 000328, loss = 0.001693
grad_step = 000329, loss = 0.001696
grad_step = 000330, loss = 0.001698
grad_step = 000331, loss = 0.001693
grad_step = 000332, loss = 0.001689
grad_step = 000333, loss = 0.001687
grad_step = 000334, loss = 0.001687
grad_step = 000335, loss = 0.001688
grad_step = 000336, loss = 0.001688
grad_step = 000337, loss = 0.001687
grad_step = 000338, loss = 0.001685
grad_step = 000339, loss = 0.001683
grad_step = 000340, loss = 0.001681
grad_step = 000341, loss = 0.001680
grad_step = 000342, loss = 0.001679
grad_step = 000343, loss = 0.001677
grad_step = 000344, loss = 0.001676
grad_step = 000345, loss = 0.001675
grad_step = 000346, loss = 0.001675
grad_step = 000347, loss = 0.001674
grad_step = 000348, loss = 0.001673
grad_step = 000349, loss = 0.001676
grad_step = 000350, loss = 0.001689
grad_step = 000351, loss = 0.001729
grad_step = 000352, loss = 0.001791
grad_step = 000353, loss = 0.001827
grad_step = 000354, loss = 0.001812
grad_step = 000355, loss = 0.001730
grad_step = 000356, loss = 0.001682
grad_step = 000357, loss = 0.001677
grad_step = 000358, loss = 0.001697
grad_step = 000359, loss = 0.001712
grad_step = 000360, loss = 0.001698
grad_step = 000361, loss = 0.001673
grad_step = 000362, loss = 0.001668
grad_step = 000363, loss = 0.001682
grad_step = 000364, loss = 0.001688
grad_step = 000365, loss = 0.001677
grad_step = 000366, loss = 0.001661
grad_step = 000367, loss = 0.001665
grad_step = 000368, loss = 0.001676
grad_step = 000369, loss = 0.001671
grad_step = 000370, loss = 0.001659
grad_step = 000371, loss = 0.001657
grad_step = 000372, loss = 0.001663
grad_step = 000373, loss = 0.001661
grad_step = 000374, loss = 0.001654
grad_step = 000375, loss = 0.001654
grad_step = 000376, loss = 0.001659
grad_step = 000377, loss = 0.001660
grad_step = 000378, loss = 0.001654
grad_step = 000379, loss = 0.001651
grad_step = 000380, loss = 0.001651
grad_step = 000381, loss = 0.001652
grad_step = 000382, loss = 0.001649
grad_step = 000383, loss = 0.001648
grad_step = 000384, loss = 0.001652
grad_step = 000385, loss = 0.001659
grad_step = 000386, loss = 0.001667
grad_step = 000387, loss = 0.001674
grad_step = 000388, loss = 0.001688
grad_step = 000389, loss = 0.001699
grad_step = 000390, loss = 0.001712
grad_step = 000391, loss = 0.001703
grad_step = 000392, loss = 0.001689
grad_step = 000393, loss = 0.001660
grad_step = 000394, loss = 0.001640
grad_step = 000395, loss = 0.001634
grad_step = 000396, loss = 0.001641
grad_step = 000397, loss = 0.001655
grad_step = 000398, loss = 0.001668
grad_step = 000399, loss = 0.001669
grad_step = 000400, loss = 0.001663
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001645
grad_step = 000402, loss = 0.001633
grad_step = 000403, loss = 0.001628
grad_step = 000404, loss = 0.001630
grad_step = 000405, loss = 0.001636
grad_step = 000406, loss = 0.001637
grad_step = 000407, loss = 0.001637
grad_step = 000408, loss = 0.001633
grad_step = 000409, loss = 0.001626
grad_step = 000410, loss = 0.001622
grad_step = 000411, loss = 0.001619
grad_step = 000412, loss = 0.001620
grad_step = 000413, loss = 0.001620
grad_step = 000414, loss = 0.001621
grad_step = 000415, loss = 0.001622
grad_step = 000416, loss = 0.001624
grad_step = 000417, loss = 0.001628
grad_step = 000418, loss = 0.001635
grad_step = 000419, loss = 0.001644
grad_step = 000420, loss = 0.001652
grad_step = 000421, loss = 0.001662
grad_step = 000422, loss = 0.001668
grad_step = 000423, loss = 0.001671
grad_step = 000424, loss = 0.001668
grad_step = 000425, loss = 0.001657
grad_step = 000426, loss = 0.001642
grad_step = 000427, loss = 0.001623
grad_step = 000428, loss = 0.001611
grad_step = 000429, loss = 0.001605
grad_step = 000430, loss = 0.001604
grad_step = 000431, loss = 0.001608
grad_step = 000432, loss = 0.001613
grad_step = 000433, loss = 0.001616
grad_step = 000434, loss = 0.001615
grad_step = 000435, loss = 0.001614
grad_step = 000436, loss = 0.001609
grad_step = 000437, loss = 0.001604
grad_step = 000438, loss = 0.001600
grad_step = 000439, loss = 0.001596
grad_step = 000440, loss = 0.001594
grad_step = 000441, loss = 0.001592
grad_step = 000442, loss = 0.001591
grad_step = 000443, loss = 0.001590
grad_step = 000444, loss = 0.001589
grad_step = 000445, loss = 0.001590
grad_step = 000446, loss = 0.001590
grad_step = 000447, loss = 0.001594
grad_step = 000448, loss = 0.001608
grad_step = 000449, loss = 0.001634
grad_step = 000450, loss = 0.001684
grad_step = 000451, loss = 0.001731
grad_step = 000452, loss = 0.001771
grad_step = 000453, loss = 0.001787
grad_step = 000454, loss = 0.001772
grad_step = 000455, loss = 0.001742
grad_step = 000456, loss = 0.001645
grad_step = 000457, loss = 0.001584
grad_step = 000458, loss = 0.001599
grad_step = 000459, loss = 0.001655
grad_step = 000460, loss = 0.001678
grad_step = 000461, loss = 0.001627
grad_step = 000462, loss = 0.001581
grad_step = 000463, loss = 0.001580
grad_step = 000464, loss = 0.001611
grad_step = 000465, loss = 0.001624
grad_step = 000466, loss = 0.001603
grad_step = 000467, loss = 0.001585
grad_step = 000468, loss = 0.001578
grad_step = 000469, loss = 0.001578
grad_step = 000470, loss = 0.001576
grad_step = 000471, loss = 0.001576
grad_step = 000472, loss = 0.001586
grad_step = 000473, loss = 0.001585
grad_step = 000474, loss = 0.001577
grad_step = 000475, loss = 0.001560
grad_step = 000476, loss = 0.001554
grad_step = 000477, loss = 0.001562
grad_step = 000478, loss = 0.001566
grad_step = 000479, loss = 0.001561
grad_step = 000480, loss = 0.001553
grad_step = 000481, loss = 0.001551
grad_step = 000482, loss = 0.001551
grad_step = 000483, loss = 0.001549
grad_step = 000484, loss = 0.001548
grad_step = 000485, loss = 0.001548
grad_step = 000486, loss = 0.001552
grad_step = 000487, loss = 0.001556
grad_step = 000488, loss = 0.001559
grad_step = 000489, loss = 0.001564
grad_step = 000490, loss = 0.001579
grad_step = 000491, loss = 0.001607
grad_step = 000492, loss = 0.001631
grad_step = 000493, loss = 0.001639
grad_step = 000494, loss = 0.001611
grad_step = 000495, loss = 0.001575
grad_step = 000496, loss = 0.001560
grad_step = 000497, loss = 0.001555
grad_step = 000498, loss = 0.001544
grad_step = 000499, loss = 0.001547
grad_step = 000500, loss = 0.001554
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001567
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

  date_run                              2020-05-24 23:20:08.072414
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.192096
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-24 23:20:08.078355
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.073921
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-24 23:20:08.085583
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.129886
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-24 23:20:08.090877
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.123256
metric_name                                             r2_score
Name: 11, dtype: object 

  


### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
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
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
100%|██████████| 10/10 [00:02<00:00,  3.46it/s, avg_epoch_loss=5.29]
INFO:root:Epoch[0] Elapsed time 2.895 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.294020
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.294019556045532 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f49742de198> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  6.81it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.469 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f49746ada90> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Fit  ####################################################### 
INFO:root:using training windows of length = 12
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in WaveNet: 97636
 30%|███       | 3/10 [00:13<00:31,  4.44s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:25<00:17,  4.29s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:36<00:04,  4.13s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:40<00:00,  4.03s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 40.287 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.860803
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.86080265045166 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f497426ea90> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:02<00:00,  4.72it/s, avg_epoch_loss=5.83]
INFO:root:Epoch[0] Elapsed time 2.117 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.832020
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.832020425796509 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4974583ac8> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [02:14<20:13, 134.87s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:18<19:55, 149.44s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [08:59<19:57, 171.07s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:30<18:17, 182.86s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [15:56<15:50, 190.02s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [19:57<13:40, 205.12s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [24:08<10:56, 218.97s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [27:39<07:13, 216.52s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [31:11<03:35, 215.30s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [35:21<00:00, 225.54s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [35:21<00:00, 212.13s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2121.348 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f49742c0080> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:02<00:00,  4.56it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 2.219 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f48f951ecc0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 35.62it/s, avg_epoch_loss=5.2]
INFO:root:Epoch[0] Elapsed time 0.282 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.196425
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.196425485610962 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f497463c8d0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} 1 validation error for MLPEncoderModel
layer_sizes
  field required (type=value_error.missing) 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/ 

                        date_run  ...            metric_name
0   2020-05-24 23:19:36.938067  ...    mean_absolute_error
1   2020-05-24 23:19:36.942087  ...     mean_squared_error
2   2020-05-24 23:19:36.945648  ...  median_absolute_error
3   2020-05-24 23:19:36.948761  ...               r2_score
4   2020-05-24 23:19:46.042007  ...    mean_absolute_error
5   2020-05-24 23:19:46.046272  ...     mean_squared_error
6   2020-05-24 23:19:46.049460  ...  median_absolute_error
7   2020-05-24 23:19:46.053424  ...               r2_score
8   2020-05-24 23:20:08.072414  ...    mean_absolute_error
9   2020-05-24 23:20:08.078355  ...     mean_squared_error
10  2020-05-24 23:20:08.085583  ...  median_absolute_error
11  2020-05-24 23:20:08.090877  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
    mpars['encoder'] = MLPEncoder()   #bug in seq2seq
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 424, in init_wrapper
    model = PydanticModel(**{**nmargs, **kwargs})
  File "pydantic/main.py", line 283, in pydantic.main.BaseModel.__init__
pydantic.error_wrappers.ValidationError: 1 validation error for MLPEncoderModel
layer_sizes
  field required (type=value_error.missing)





 ************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt

  dataset/json/benchmark.json 

  Custom benchmark 

  ['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'] 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_timeseries/test01/ 

  Model List [{'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}] 

  


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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fadf9fe2ef0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355159.3438
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 254631.9219
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 149965.3125
Epoch 4/10

1/1 [==============================] - 0s 123ms/step - loss: 69603.4141
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 33129.3984
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 18093.0547
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 11140.3555
Epoch 8/10

1/1 [==============================] - 0s 90ms/step - loss: 7520.6914
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 5476.1899
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 4244.0474

  #### Inference Need return ypred, ytrue ######################### 
[[ 7.58241236e-01  1.00474930e+01  8.58631516e+00  8.28756237e+00
   1.07363319e+01  1.10237722e+01  1.01954842e+01  1.19974575e+01
   1.02631092e+01  1.08429861e+01  9.71825123e+00  1.23190956e+01
   1.03930454e+01  1.03556147e+01  9.58613873e+00  9.75504303e+00
   9.51171398e+00  1.06540012e+01  1.16401415e+01  8.92627048e+00
   1.03619499e+01  1.00662251e+01  1.04804764e+01  1.16749897e+01
   1.14299002e+01  1.08704624e+01  9.63304615e+00  1.12094498e+01
   1.12739305e+01  1.06485205e+01  1.16434851e+01  1.15409822e+01
   1.05735712e+01  1.03733549e+01  1.24620571e+01  1.08848181e+01
   1.05585346e+01  1.03585634e+01  8.18815041e+00  9.75719738e+00
   1.10162687e+01  9.81094933e+00  8.52575207e+00  1.12402344e+01
   1.12881069e+01  9.41869450e+00  1.02830753e+01  1.10104246e+01
   1.19105206e+01  1.03413849e+01  1.16141424e+01  1.11168051e+01
   1.19663343e+01  1.06491795e+01  1.23195705e+01  9.21172810e+00
   1.15085735e+01  1.22995043e+01  9.15082741e+00  8.33281612e+00
   1.05767429e+00  5.35720825e-01 -1.23336005e+00  6.71170533e-01
   5.35058081e-01 -1.44796252e+00  1.05740666e-01 -1.53557062e+00
  -8.17293525e-01 -7.97613561e-01  1.45518100e+00 -2.00497293e+00
   7.60752797e-01 -4.44297463e-01  3.96611914e-02 -2.59984016e-01
  -2.87828875e+00  1.19433320e+00  6.52335763e-01 -2.12763619e+00
   4.98559117e-01  2.11275339e+00 -1.15015638e+00 -7.58543015e-01
  -4.15328383e-01  1.95329106e+00  7.59741664e-02  1.21132195e-01
   3.27013023e-02 -6.58090472e-01 -3.79569113e-01  3.10168535e-01
   1.12337852e+00  4.51684892e-01 -2.07399583e+00 -2.59040689e+00
  -1.08172178e-01 -1.17318964e+00 -1.31538510e+00  6.73461795e-01
   3.62629116e-01  1.18733668e+00 -2.03127551e+00  4.42589104e-01
  -2.74532294e+00  2.99029976e-01 -4.21253622e-01  4.08789039e-01
   2.83518970e-01  1.58620048e+00  1.27533817e+00 -2.86138368e+00
  -4.36971188e-01 -3.25132251e-01  1.26318741e+00  1.99838495e+00
   1.35874331e+00  1.31342649e+00 -1.80389225e+00 -1.41349643e-01
   1.62292290e+00  4.14418578e-02 -2.20659447e+00 -4.96412337e-01
   1.28560710e+00  1.55887258e+00  6.95712030e-01  1.22685122e+00
  -6.55932009e-01  2.91936517e-01  1.10033178e+00  5.34958303e-01
   1.65645629e-01  3.73533517e-02 -1.15089452e+00  5.53817868e-01
   3.26676846e-01  6.56691492e-01  2.67111611e+00  7.65699387e-01
   1.05245507e+00 -1.24764200e-02  4.25621778e-01 -7.11140394e-01
   3.06997120e-01  1.07459259e+00  1.15023673e-01  4.20615017e-01
   1.53063321e+00  5.04981220e-01 -3.72798800e-01  1.60106826e+00
  -1.34615934e+00 -1.04530156e+00 -1.50864029e+00 -5.80288529e-01
   1.15972471e+00  9.45361614e-01  1.49709320e+00  1.20632482e+00
   2.39588737e-01  9.79952991e-01  8.57984304e-01  1.65162563e+00
  -2.43692130e-01 -5.44475853e-01  1.04109907e+00 -1.06775105e-01
   1.40435410e+00  8.10143650e-01  1.07438874e+00  9.05665815e-01
  -8.11538100e-03  7.89394379e-02 -3.51867080e-02  1.87916231e+00
   7.34012485e-01  6.90618932e-01 -2.14371872e+00 -4.43495095e-01
   5.59347808e-01  1.02639685e+01  8.47668934e+00  1.28643312e+01
   9.11203480e+00  1.00661182e+01  1.09180727e+01  1.12446222e+01
   1.16087208e+01  9.58287048e+00  1.09429092e+01  1.03207226e+01
   1.07105751e+01  1.23705750e+01  1.00280838e+01  1.12182093e+01
   8.23085499e+00  1.19493446e+01  9.54344463e+00  1.06621246e+01
   1.07557125e+01  1.01473942e+01  8.99940491e+00  1.06641283e+01
   1.00920010e+01  1.16750059e+01  1.07213955e+01  1.16015892e+01
   9.52057362e+00  1.19284277e+01  9.48011875e+00  1.16728449e+01
   1.24892349e+01  9.57375717e+00  1.10168743e+01  1.25381212e+01
   1.25771179e+01  8.57304764e+00  1.11344728e+01  1.09794016e+01
   7.22049618e+00  9.38496780e+00  1.12940397e+01  1.13727760e+01
   1.00391150e+01  1.06867962e+01  1.08547974e+01  1.15774374e+01
   1.08296547e+01  9.97325897e+00  8.34868240e+00  1.03506031e+01
   1.14250660e+01  1.04032898e+01  1.03991299e+01  1.14823914e+01
   1.10827360e+01  1.06450348e+01  1.23603506e+01  1.21311255e+01
   2.51233876e-01  1.10382617e-01  1.05965638e+00  3.39905977e+00
   2.00107098e-01  2.69608450e+00  1.04518545e+00  1.02941990e+00
   3.81949663e-01  3.23044348e+00  5.34855127e-02  5.37047327e-01
   2.07081556e-01  3.47299039e-01  2.25259662e-01  4.04028225e+00
   1.63387132e+00  3.27896070e+00  7.52836108e-01  1.73452151e+00
   3.03526521e-01  1.96390033e-01  3.38096023e-02  2.29405403e+00
   5.51586449e-01  3.28032076e-01  5.02126396e-01  1.81043100e+00
   4.73028064e-01  2.41983867e+00  2.21140087e-01  1.33261418e+00
   3.50891531e-01  2.18965530e+00  1.57464600e+00  1.50206256e+00
   2.26130152e+00  2.11918414e-01  4.36107159e-01  5.52569866e-01
   1.89825809e+00  2.20164418e-01  3.25451279e+00  1.75449085e+00
   4.77095008e-01  2.42477596e-01  8.96132767e-01  2.61025047e+00
   1.06030536e+00  1.91480350e+00  1.84638262e-01  4.78353918e-01
   6.18922770e-01  9.77392197e-02  8.28112841e-01  7.47855365e-01
   1.27590942e+00  4.80714858e-01  2.37146854e-01  3.91427159e-01
   3.05290556e+00  1.53665185e+00  2.94672072e-01  3.34888935e-01
   8.53613853e-01  3.73989105e-01  7.69696474e-01  5.66570401e-01
   2.07964897e+00  4.44297612e-01  1.11395597e-01  1.10858846e+00
   7.40005374e-02  5.01876593e-01  5.57609141e-01  1.85073781e+00
   2.67482424e+00  2.32530260e+00  1.71122873e+00  2.79887390e+00
   2.55586505e-01  3.89538527e-01  5.22809684e-01  2.57085013e+00
   1.78458452e-01  3.70021999e-01  5.09395659e-01  7.36777961e-01
   8.67535114e-01  3.44021857e-01  2.21316040e-01  2.17744970e+00
   3.33835006e-01  1.37909615e+00  2.43881464e+00  1.10485506e+00
   1.24785411e+00  2.92233038e+00  2.61524868e+00  2.67013311e+00
   3.61222029e-01  2.88335657e+00  1.61220765e+00  1.03120399e+00
   4.63697433e-01  2.78601170e-01  3.95328760e-01  5.68162858e-01
   1.98485184e+00  1.39184666e+00  1.48609030e+00  1.92544127e+00
   2.02802610e+00  1.45900893e+00  1.35532808e+00  1.39993834e+00
   1.11272216e+00  8.03375006e-01  3.41547668e-01  1.21652758e+00
   8.60133266e+00 -6.90220356e+00 -5.67612171e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 23:56:53.492233
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    91.114
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-24 23:56:53.499298
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8334.62
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-24 23:56:53.504516
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   91.2444
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-24 23:56:53.509338
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -745.424
metric_name                                             r2_score
Name: 3, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140385307162328
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405934272
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405934776
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405529840
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405530344
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140382405530848

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fadf9fd9080> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.543901
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.509596
grad_step = 000002, loss = 0.486128
grad_step = 000003, loss = 0.461945
grad_step = 000004, loss = 0.434883
grad_step = 000005, loss = 0.405255
grad_step = 000006, loss = 0.375870
grad_step = 000007, loss = 0.351548
grad_step = 000008, loss = 0.335632
grad_step = 000009, loss = 0.317571
grad_step = 000010, loss = 0.297870
grad_step = 000011, loss = 0.280454
grad_step = 000012, loss = 0.267626
grad_step = 000013, loss = 0.257748
grad_step = 000014, loss = 0.248483
grad_step = 000015, loss = 0.238800
grad_step = 000016, loss = 0.228886
grad_step = 000017, loss = 0.218523
grad_step = 000018, loss = 0.207786
grad_step = 000019, loss = 0.197174
grad_step = 000020, loss = 0.187318
grad_step = 000021, loss = 0.178344
grad_step = 000022, loss = 0.169370
grad_step = 000023, loss = 0.160377
grad_step = 000024, loss = 0.151878
grad_step = 000025, loss = 0.143956
grad_step = 000026, loss = 0.136446
grad_step = 000027, loss = 0.129055
grad_step = 000028, loss = 0.121939
grad_step = 000029, loss = 0.115279
grad_step = 000030, loss = 0.108860
grad_step = 000031, loss = 0.102476
grad_step = 000032, loss = 0.096147
grad_step = 000033, loss = 0.090252
grad_step = 000034, loss = 0.084812
grad_step = 000035, loss = 0.079522
grad_step = 000036, loss = 0.074395
grad_step = 000037, loss = 0.069611
grad_step = 000038, loss = 0.065209
grad_step = 000039, loss = 0.060989
grad_step = 000040, loss = 0.056886
grad_step = 000041, loss = 0.053009
grad_step = 000042, loss = 0.048856
grad_step = 000043, loss = 0.044285
grad_step = 000044, loss = 0.040204
grad_step = 000045, loss = 0.037310
grad_step = 000046, loss = 0.034842
grad_step = 000047, loss = 0.032250
grad_step = 000048, loss = 0.029536
grad_step = 000049, loss = 0.026832
grad_step = 000050, loss = 0.024463
grad_step = 000051, loss = 0.022438
grad_step = 000052, loss = 0.020457
grad_step = 000053, loss = 0.018768
grad_step = 000054, loss = 0.017197
grad_step = 000055, loss = 0.015583
grad_step = 000056, loss = 0.014118
grad_step = 000057, loss = 0.012792
grad_step = 000058, loss = 0.011639
grad_step = 000059, loss = 0.010649
grad_step = 000060, loss = 0.009671
grad_step = 000061, loss = 0.008809
grad_step = 000062, loss = 0.008020
grad_step = 000063, loss = 0.007280
grad_step = 000064, loss = 0.006658
grad_step = 000065, loss = 0.006059
grad_step = 000066, loss = 0.005526
grad_step = 000067, loss = 0.005086
grad_step = 000068, loss = 0.004690
grad_step = 000069, loss = 0.004330
grad_step = 000070, loss = 0.003987
grad_step = 000071, loss = 0.003689
grad_step = 000072, loss = 0.003448
grad_step = 000073, loss = 0.003218
grad_step = 000074, loss = 0.003034
grad_step = 000075, loss = 0.002869
grad_step = 000076, loss = 0.002728
grad_step = 000077, loss = 0.002614
grad_step = 000078, loss = 0.002500
grad_step = 000079, loss = 0.002404
grad_step = 000080, loss = 0.002327
grad_step = 000081, loss = 0.002278
grad_step = 000082, loss = 0.002235
grad_step = 000083, loss = 0.002189
grad_step = 000084, loss = 0.002153
grad_step = 000085, loss = 0.002122
grad_step = 000086, loss = 0.002104
grad_step = 000087, loss = 0.002090
grad_step = 000088, loss = 0.002078
grad_step = 000089, loss = 0.002068
grad_step = 000090, loss = 0.002059
grad_step = 000091, loss = 0.002050
grad_step = 000092, loss = 0.002040
grad_step = 000093, loss = 0.002035
grad_step = 000094, loss = 0.002030
grad_step = 000095, loss = 0.002024
grad_step = 000096, loss = 0.002017
grad_step = 000097, loss = 0.002009
grad_step = 000098, loss = 0.002006
grad_step = 000099, loss = 0.002009
grad_step = 000100, loss = 0.002029
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002056
grad_step = 000102, loss = 0.002063
grad_step = 000103, loss = 0.001998
grad_step = 000104, loss = 0.001955
grad_step = 000105, loss = 0.001977
grad_step = 000106, loss = 0.001994
grad_step = 000107, loss = 0.001959
grad_step = 000108, loss = 0.001923
grad_step = 000109, loss = 0.001934
grad_step = 000110, loss = 0.001949
grad_step = 000111, loss = 0.001923
grad_step = 000112, loss = 0.001892
grad_step = 000113, loss = 0.001892
grad_step = 000114, loss = 0.001905
grad_step = 000115, loss = 0.001900
grad_step = 000116, loss = 0.001874
grad_step = 000117, loss = 0.001856
grad_step = 000118, loss = 0.001859
grad_step = 000119, loss = 0.001865
grad_step = 000120, loss = 0.001858
grad_step = 000121, loss = 0.001840
grad_step = 000122, loss = 0.001830
grad_step = 000123, loss = 0.001828
grad_step = 000124, loss = 0.001829
grad_step = 000125, loss = 0.001829
grad_step = 000126, loss = 0.001827
grad_step = 000127, loss = 0.001818
grad_step = 000128, loss = 0.001807
grad_step = 000129, loss = 0.001798
grad_step = 000130, loss = 0.001794
grad_step = 000131, loss = 0.001793
grad_step = 000132, loss = 0.001792
grad_step = 000133, loss = 0.001793
grad_step = 000134, loss = 0.001798
grad_step = 000135, loss = 0.001808
grad_step = 000136, loss = 0.001820
grad_step = 000137, loss = 0.001832
grad_step = 000138, loss = 0.001828
grad_step = 000139, loss = 0.001809
grad_step = 000140, loss = 0.001774
grad_step = 000141, loss = 0.001756
grad_step = 000142, loss = 0.001761
grad_step = 000143, loss = 0.001776
grad_step = 000144, loss = 0.001789
grad_step = 000145, loss = 0.001787
grad_step = 000146, loss = 0.001769
grad_step = 000147, loss = 0.001746
grad_step = 000148, loss = 0.001731
grad_step = 000149, loss = 0.001728
grad_step = 000150, loss = 0.001734
grad_step = 000151, loss = 0.001742
grad_step = 000152, loss = 0.001749
grad_step = 000153, loss = 0.001755
grad_step = 000154, loss = 0.001745
grad_step = 000155, loss = 0.001732
grad_step = 000156, loss = 0.001710
grad_step = 000157, loss = 0.001699
grad_step = 000158, loss = 0.001700
grad_step = 000159, loss = 0.001701
grad_step = 000160, loss = 0.001703
grad_step = 000161, loss = 0.001704
grad_step = 000162, loss = 0.001712
grad_step = 000163, loss = 0.001726
grad_step = 000164, loss = 0.001720
grad_step = 000165, loss = 0.001706
grad_step = 000166, loss = 0.001683
grad_step = 000167, loss = 0.001675
grad_step = 000168, loss = 0.001683
grad_step = 000169, loss = 0.001694
grad_step = 000170, loss = 0.001712
grad_step = 000171, loss = 0.001692
grad_step = 000172, loss = 0.001668
grad_step = 000173, loss = 0.001648
grad_step = 000174, loss = 0.001649
grad_step = 000175, loss = 0.001663
grad_step = 000176, loss = 0.001663
grad_step = 000177, loss = 0.001656
grad_step = 000178, loss = 0.001648
grad_step = 000179, loss = 0.001651
grad_step = 000180, loss = 0.001662
grad_step = 000181, loss = 0.001683
grad_step = 000182, loss = 0.001704
grad_step = 000183, loss = 0.001699
grad_step = 000184, loss = 0.001677
grad_step = 000185, loss = 0.001640
grad_step = 000186, loss = 0.001622
grad_step = 000187, loss = 0.001631
grad_step = 000188, loss = 0.001641
grad_step = 000189, loss = 0.001645
grad_step = 000190, loss = 0.001638
grad_step = 000191, loss = 0.001630
grad_step = 000192, loss = 0.001628
grad_step = 000193, loss = 0.001622
grad_step = 000194, loss = 0.001620
grad_step = 000195, loss = 0.001616
grad_step = 000196, loss = 0.001614
grad_step = 000197, loss = 0.001613
grad_step = 000198, loss = 0.001614
grad_step = 000199, loss = 0.001616
grad_step = 000200, loss = 0.001614
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001610
grad_step = 000202, loss = 0.001602
grad_step = 000203, loss = 0.001595
grad_step = 000204, loss = 0.001589
grad_step = 000205, loss = 0.001587
grad_step = 000206, loss = 0.001586
grad_step = 000207, loss = 0.001587
grad_step = 000208, loss = 0.001589
grad_step = 000209, loss = 0.001592
grad_step = 000210, loss = 0.001596
grad_step = 000211, loss = 0.001600
grad_step = 000212, loss = 0.001607
grad_step = 000213, loss = 0.001613
grad_step = 000214, loss = 0.001625
grad_step = 000215, loss = 0.001632
grad_step = 000216, loss = 0.001658
grad_step = 000217, loss = 0.001646
grad_step = 000218, loss = 0.001630
grad_step = 000219, loss = 0.001580
grad_step = 000220, loss = 0.001584
grad_step = 000221, loss = 0.001623
grad_step = 000222, loss = 0.001614
grad_step = 000223, loss = 0.001586
grad_step = 000224, loss = 0.001577
grad_step = 000225, loss = 0.001594
grad_step = 000226, loss = 0.001615
grad_step = 000227, loss = 0.001598
grad_step = 000228, loss = 0.001576
grad_step = 000229, loss = 0.001554
grad_step = 000230, loss = 0.001555
grad_step = 000231, loss = 0.001570
grad_step = 000232, loss = 0.001571
grad_step = 000233, loss = 0.001560
grad_step = 000234, loss = 0.001548
grad_step = 000235, loss = 0.001551
grad_step = 000236, loss = 0.001564
grad_step = 000237, loss = 0.001569
grad_step = 000238, loss = 0.001571
grad_step = 000239, loss = 0.001575
grad_step = 000240, loss = 0.001603
grad_step = 000241, loss = 0.001636
grad_step = 000242, loss = 0.001671
grad_step = 000243, loss = 0.001645
grad_step = 000244, loss = 0.001589
grad_step = 000245, loss = 0.001537
grad_step = 000246, loss = 0.001547
grad_step = 000247, loss = 0.001589
grad_step = 000248, loss = 0.001590
grad_step = 000249, loss = 0.001558
grad_step = 000250, loss = 0.001529
grad_step = 000251, loss = 0.001538
grad_step = 000252, loss = 0.001563
grad_step = 000253, loss = 0.001561
grad_step = 000254, loss = 0.001539
grad_step = 000255, loss = 0.001519
grad_step = 000256, loss = 0.001522
grad_step = 000257, loss = 0.001535
grad_step = 000258, loss = 0.001537
grad_step = 000259, loss = 0.001526
grad_step = 000260, loss = 0.001513
grad_step = 000261, loss = 0.001509
grad_step = 000262, loss = 0.001515
grad_step = 000263, loss = 0.001522
grad_step = 000264, loss = 0.001523
grad_step = 000265, loss = 0.001518
grad_step = 000266, loss = 0.001514
grad_step = 000267, loss = 0.001520
grad_step = 000268, loss = 0.001541
grad_step = 000269, loss = 0.001599
grad_step = 000270, loss = 0.001640
grad_step = 000271, loss = 0.001716
grad_step = 000272, loss = 0.001619
grad_step = 000273, loss = 0.001565
grad_step = 000274, loss = 0.001568
grad_step = 000275, loss = 0.001537
grad_step = 000276, loss = 0.001522
grad_step = 000277, loss = 0.001554
grad_step = 000278, loss = 0.001569
grad_step = 000279, loss = 0.001542
grad_step = 000280, loss = 0.001508
grad_step = 000281, loss = 0.001520
grad_step = 000282, loss = 0.001539
grad_step = 000283, loss = 0.001531
grad_step = 000284, loss = 0.001493
grad_step = 000285, loss = 0.001480
grad_step = 000286, loss = 0.001503
grad_step = 000287, loss = 0.001515
grad_step = 000288, loss = 0.001503
grad_step = 000289, loss = 0.001483
grad_step = 000290, loss = 0.001486
grad_step = 000291, loss = 0.001497
grad_step = 000292, loss = 0.001491
grad_step = 000293, loss = 0.001474
grad_step = 000294, loss = 0.001467
grad_step = 000295, loss = 0.001473
grad_step = 000296, loss = 0.001480
grad_step = 000297, loss = 0.001474
grad_step = 000298, loss = 0.001464
grad_step = 000299, loss = 0.001458
grad_step = 000300, loss = 0.001461
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001465
grad_step = 000302, loss = 0.001463
grad_step = 000303, loss = 0.001456
grad_step = 000304, loss = 0.001451
grad_step = 000305, loss = 0.001451
grad_step = 000306, loss = 0.001454
grad_step = 000307, loss = 0.001453
grad_step = 000308, loss = 0.001449
grad_step = 000309, loss = 0.001445
grad_step = 000310, loss = 0.001445
grad_step = 000311, loss = 0.001447
grad_step = 000312, loss = 0.001450
grad_step = 000313, loss = 0.001453
grad_step = 000314, loss = 0.001463
grad_step = 000315, loss = 0.001478
grad_step = 000316, loss = 0.001518
grad_step = 000317, loss = 0.001539
grad_step = 000318, loss = 0.001576
grad_step = 000319, loss = 0.001540
grad_step = 000320, loss = 0.001494
grad_step = 000321, loss = 0.001450
grad_step = 000322, loss = 0.001442
grad_step = 000323, loss = 0.001465
grad_step = 000324, loss = 0.001489
grad_step = 000325, loss = 0.001496
grad_step = 000326, loss = 0.001467
grad_step = 000327, loss = 0.001439
grad_step = 000328, loss = 0.001430
grad_step = 000329, loss = 0.001436
grad_step = 000330, loss = 0.001437
grad_step = 000331, loss = 0.001427
grad_step = 000332, loss = 0.001419
grad_step = 000333, loss = 0.001425
grad_step = 000334, loss = 0.001441
grad_step = 000335, loss = 0.001453
grad_step = 000336, loss = 0.001466
grad_step = 000337, loss = 0.001468
grad_step = 000338, loss = 0.001485
grad_step = 000339, loss = 0.001495
grad_step = 000340, loss = 0.001508
grad_step = 000341, loss = 0.001481
grad_step = 000342, loss = 0.001442
grad_step = 000343, loss = 0.001409
grad_step = 000344, loss = 0.001406
grad_step = 000345, loss = 0.001420
grad_step = 000346, loss = 0.001430
grad_step = 000347, loss = 0.001432
grad_step = 000348, loss = 0.001425
grad_step = 000349, loss = 0.001419
grad_step = 000350, loss = 0.001405
grad_step = 000351, loss = 0.001393
grad_step = 000352, loss = 0.001389
grad_step = 000353, loss = 0.001396
grad_step = 000354, loss = 0.001406
grad_step = 000355, loss = 0.001409
grad_step = 000356, loss = 0.001410
grad_step = 000357, loss = 0.001407
grad_step = 000358, loss = 0.001410
grad_step = 000359, loss = 0.001408
grad_step = 000360, loss = 0.001405
grad_step = 000361, loss = 0.001395
grad_step = 000362, loss = 0.001386
grad_step = 000363, loss = 0.001380
grad_step = 000364, loss = 0.001377
grad_step = 000365, loss = 0.001374
grad_step = 000366, loss = 0.001370
grad_step = 000367, loss = 0.001367
grad_step = 000368, loss = 0.001366
grad_step = 000369, loss = 0.001367
grad_step = 000370, loss = 0.001369
grad_step = 000371, loss = 0.001372
grad_step = 000372, loss = 0.001377
grad_step = 000373, loss = 0.001389
grad_step = 000374, loss = 0.001411
grad_step = 000375, loss = 0.001457
grad_step = 000376, loss = 0.001512
grad_step = 000377, loss = 0.001601
grad_step = 000378, loss = 0.001617
grad_step = 000379, loss = 0.001606
grad_step = 000380, loss = 0.001473
grad_step = 000381, loss = 0.001371
grad_step = 000382, loss = 0.001363
grad_step = 000383, loss = 0.001432
grad_step = 000384, loss = 0.001488
grad_step = 000385, loss = 0.001446
grad_step = 000386, loss = 0.001383
grad_step = 000387, loss = 0.001343
grad_step = 000388, loss = 0.001358
grad_step = 000389, loss = 0.001403
grad_step = 000390, loss = 0.001423
grad_step = 000391, loss = 0.001426
grad_step = 000392, loss = 0.001386
grad_step = 000393, loss = 0.001350
grad_step = 000394, loss = 0.001333
grad_step = 000395, loss = 0.001340
grad_step = 000396, loss = 0.001362
grad_step = 000397, loss = 0.001373
grad_step = 000398, loss = 0.001370
grad_step = 000399, loss = 0.001349
grad_step = 000400, loss = 0.001329
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001320
grad_step = 000402, loss = 0.001323
grad_step = 000403, loss = 0.001331
grad_step = 000404, loss = 0.001339
grad_step = 000405, loss = 0.001344
grad_step = 000406, loss = 0.001343
grad_step = 000407, loss = 0.001339
grad_step = 000408, loss = 0.001331
grad_step = 000409, loss = 0.001322
grad_step = 000410, loss = 0.001313
grad_step = 000411, loss = 0.001306
grad_step = 000412, loss = 0.001303
grad_step = 000413, loss = 0.001302
grad_step = 000414, loss = 0.001303
grad_step = 000415, loss = 0.001304
grad_step = 000416, loss = 0.001308
grad_step = 000417, loss = 0.001313
grad_step = 000418, loss = 0.001322
grad_step = 000419, loss = 0.001334
grad_step = 000420, loss = 0.001356
grad_step = 000421, loss = 0.001381
grad_step = 000422, loss = 0.001424
grad_step = 000423, loss = 0.001454
grad_step = 000424, loss = 0.001490
grad_step = 000425, loss = 0.001460
grad_step = 000426, loss = 0.001405
grad_step = 000427, loss = 0.001324
grad_step = 000428, loss = 0.001283
grad_step = 000429, loss = 0.001300
grad_step = 000430, loss = 0.001344
grad_step = 000431, loss = 0.001378
grad_step = 000432, loss = 0.001366
grad_step = 000433, loss = 0.001337
grad_step = 000434, loss = 0.001297
grad_step = 000435, loss = 0.001274
grad_step = 000436, loss = 0.001272
grad_step = 000437, loss = 0.001286
grad_step = 000438, loss = 0.001308
grad_step = 000439, loss = 0.001323
grad_step = 000440, loss = 0.001333
grad_step = 000441, loss = 0.001330
grad_step = 000442, loss = 0.001321
grad_step = 000443, loss = 0.001304
grad_step = 000444, loss = 0.001289
grad_step = 000445, loss = 0.001271
grad_step = 000446, loss = 0.001259
grad_step = 000447, loss = 0.001252
grad_step = 000448, loss = 0.001251
grad_step = 000449, loss = 0.001255
grad_step = 000450, loss = 0.001262
grad_step = 000451, loss = 0.001272
grad_step = 000452, loss = 0.001283
grad_step = 000453, loss = 0.001304
grad_step = 000454, loss = 0.001328
grad_step = 000455, loss = 0.001370
grad_step = 000456, loss = 0.001404
grad_step = 000457, loss = 0.001442
grad_step = 000458, loss = 0.001423
grad_step = 000459, loss = 0.001379
grad_step = 000460, loss = 0.001297
grad_step = 000461, loss = 0.001246
grad_step = 000462, loss = 0.001242
grad_step = 000463, loss = 0.001275
grad_step = 000464, loss = 0.001314
grad_step = 000465, loss = 0.001323
grad_step = 000466, loss = 0.001309
grad_step = 000467, loss = 0.001268
grad_step = 000468, loss = 0.001235
grad_step = 000469, loss = 0.001221
grad_step = 000470, loss = 0.001225
grad_step = 000471, loss = 0.001241
grad_step = 000472, loss = 0.001260
grad_step = 000473, loss = 0.001283
grad_step = 000474, loss = 0.001299
grad_step = 000475, loss = 0.001316
grad_step = 000476, loss = 0.001316
grad_step = 000477, loss = 0.001306
grad_step = 000478, loss = 0.001275
grad_step = 000479, loss = 0.001240
grad_step = 000480, loss = 0.001213
grad_step = 000481, loss = 0.001204
grad_step = 000482, loss = 0.001212
grad_step = 000483, loss = 0.001229
grad_step = 000484, loss = 0.001245
grad_step = 000485, loss = 0.001252
grad_step = 000486, loss = 0.001256
grad_step = 000487, loss = 0.001246
grad_step = 000488, loss = 0.001238
grad_step = 000489, loss = 0.001221
grad_step = 000490, loss = 0.001209
grad_step = 000491, loss = 0.001197
grad_step = 000492, loss = 0.001189
grad_step = 000493, loss = 0.001184
grad_step = 000494, loss = 0.001183
grad_step = 000495, loss = 0.001184
grad_step = 000496, loss = 0.001186
grad_step = 000497, loss = 0.001191
grad_step = 000498, loss = 0.001200
grad_step = 000499, loss = 0.001218
grad_step = 000500, loss = 0.001249
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001307
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

  date_run                              2020-05-24 23:57:16.974005
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.24937
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-24 23:57:16.980524
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.16186
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-24 23:57:16.987277
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.13814
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-24 23:57:16.992471
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.45953
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fad536bac88> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 23:57:35.559578
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-24 23:57:35.564376
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-24 23:57:35.568922
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-24 23:57:35.572616
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
0   2020-05-24 23:56:53.492233  ...    mean_absolute_error
1   2020-05-24 23:56:53.499298  ...     mean_squared_error
2   2020-05-24 23:56:53.504516  ...  median_absolute_error
3   2020-05-24 23:56:53.509338  ...               r2_score
4   2020-05-24 23:57:16.974005  ...    mean_absolute_error
5   2020-05-24 23:57:16.980524  ...     mean_squared_error
6   2020-05-24 23:57:16.987277  ...  median_absolute_error
7   2020-05-24 23:57:16.992471  ...               r2_score
8   2020-05-24 23:57:35.559578  ...    mean_absolute_error
9   2020-05-24 23:57:35.564376  ...     mean_squared_error
10  2020-05-24 23:57:35.568922  ...  median_absolute_error
11  2020-05-24 23:57:35.572616  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 118, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

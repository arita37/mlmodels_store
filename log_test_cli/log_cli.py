
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

  ##### Fit <mlmodels.model_tf.1_lstm.Model object at 0x7fc8ea882cf8> 
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

  ##### Save <tensorflow.python.client.session.Session object at 0x7fc8b7d91fd0> 
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

  ##### Predict: <tensorflow.python.client.session.Session object at 0x7f09851976a0> 
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fe7b4c9b160> 

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
 [-0.12860131 -0.05081505  0.00883749  0.01363465 -0.02189918 -0.18338561]
 [-0.23603639  0.03653605  0.02657127  0.18496251  0.02358393  0.16293697]
 [-0.00370685  0.06840706 -0.13776301  0.27094099 -0.06576131 -0.04348953]
 [-0.14067619  0.10684681  0.01820574  0.33744302  0.04719005  0.15715533]
 [ 0.31323785  0.16740167 -0.08590227 -0.0829576  -0.18214467 -0.35788524]
 [ 0.39307272 -0.19700579  0.22607911  0.48555118 -0.41028085 -0.12886941]
 [ 0.32905549 -0.1926614  -0.05473661  0.26421613  0.66294247  0.21110836]
 [ 0.19797207 -0.10359257  0.25057775  0.68145102 -0.28204763  0.45024574]
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
{'loss': 0.4322054050862789, 'loss_history': []}

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
{'loss': 0.3312234878540039, 'loss_history': []}

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

  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f930bda6198> 

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
 [ 0.0101859   0.08031023  0.03708758  0.04294459 -0.06634052  0.01820028]
 [ 0.14012188 -0.04065072 -0.06180897 -0.10262728  0.09237104 -0.07001469]
 [ 0.12576672 -0.01082949 -0.00510307  0.0579341  -0.08804878 -0.08768222]
 [ 0.21565449 -0.13007648 -0.0731608  -0.18240929  0.33387303 -0.16108447]
 [ 0.32663858  0.09670994  0.25469664  0.6414476  -0.2212217  -0.29833877]
 [ 0.03743885  0.06161596  0.03756379  0.30604303  0.17939989 -0.10568619]
 [ 0.13654783 -0.5963757  -0.13653566  0.23963422 -0.16713725  0.2075073 ]
 [ 0.73153484  0.08710942 -0.66431874 -0.11803953  0.72087246 -0.06709679]
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
{'loss': 0.4795107841491699, 'loss_history': []}

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
{'loss': 0.39132798835635185, 'loss_history': []}

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
[32m[I 2020-05-25 23:19:26,584][0m Finished trial#0 resulted in value: 2.1581093221902847. Current best value is 2.1581093221902847 with parameters: {'learning_rate': 0.0529432096831136, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-25 23:19:27,896][0m Finished trial#1 resulted in value: 8.8061203956604. Current best value is 2.1581093221902847 with parameters: {'learning_rate': 0.0529432096831136, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.0529432096831136, 'num_layers': 3, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 5, 'epoch': 2, 'best_value': 2.1581093221902847, 'model_name': None} 





 ************************************************************************************************************************
ml_optim --do test   --model_uri model_tf.1_lstm   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-25 23:19:35,023][0m Finished trial#0 resulted in value: 14.824942111968994. Current best value is 14.824942111968994 with parameters: {'learning_rate': 0.0848817331099201, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-25 23:19:36,369][0m Finished trial#1 resulted in value: 0.7862761914730072. Current best value is 0.7862761914730072 with parameters: {'learning_rate': 0.019077804291149435, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa951709eb8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 23:19:54.704345
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-25 23:19:54.708640
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-25 23:19:54.712262
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-25 23:19:54.715798
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa940d24080> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 359582.6875
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 277368.4375
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 174684.5938
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 103928.4922
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 61279.0195
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 36893.2422
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 23025.3633
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 15034.8623
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 10642.5527
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 8097.7173

  #### Inference Need return ypred, ytrue ######################### 
[[-8.39751214e-03  6.71400690e+00  6.33326292e+00  7.85244989e+00
   7.93026781e+00  5.74289370e+00  6.32038450e+00  5.88292742e+00
   7.55585194e+00  6.16864872e+00  8.07298470e+00  5.94067907e+00
   6.31272221e+00  6.06142092e+00  7.04356909e+00  7.02432823e+00
   8.29759598e+00  7.78452826e+00  8.11949635e+00  7.54326868e+00
   6.75284147e+00  6.95194340e+00  6.89327383e+00  7.64055490e+00
   6.67392445e+00  6.01579952e+00  7.46034384e+00  6.67370939e+00
   6.31731606e+00  5.97897291e+00  8.06626987e+00  6.25019121e+00
   7.57338715e+00  8.43972588e+00  7.39123058e+00  8.03287697e+00
   8.10601044e+00  5.94035816e+00  6.10580921e+00  7.04443216e+00
   8.16332722e+00  7.91944838e+00  8.35736847e+00  9.01438808e+00
   7.25673103e+00  8.90853119e+00  6.67184830e+00  7.77821207e+00
   7.01709080e+00  6.94863796e+00  8.51073170e+00  7.74656630e+00
   7.67579651e+00  8.40626812e+00  7.65319681e+00  6.61596823e+00
   7.14158964e+00  8.81519890e+00  8.30860615e+00  8.00185680e+00
  -1.85612559e+00 -1.41707289e+00 -1.97878420e-01  6.94985509e-01
  -2.28029206e-01 -8.39547634e-01  1.68131042e+00  4.36027527e-01
   5.04157841e-01 -2.64915645e-01  1.00360560e+00  1.16319954e-01
   2.01691961e+00 -3.28808308e-01  9.22618270e-01  5.59979856e-01
   1.82322454e+00  9.86233205e-02 -1.78914464e+00 -6.28711939e-01
  -6.46996975e-01 -5.62678277e-01 -1.20168495e+00 -1.32831705e+00
   1.15565801e+00 -1.23307109e+00 -2.37379646e+00  8.05852652e-01
   2.17015088e-01 -2.96582550e-01  5.63698113e-02  3.41975600e-01
  -1.25636399e+00  3.38882416e-01  9.40066576e-03 -1.59577087e-01
  -1.29662967e+00 -5.08468091e-01  9.85889792e-01 -4.78585839e-01
   9.40113783e-01  1.38861883e+00  2.02460289e-01 -2.07401752e+00
   7.58869410e-01  5.91513991e-01  1.28919706e-01  1.41381979e-01
  -6.50947332e-01 -5.89771032e-01 -3.71894151e-01  8.03284496e-02
  -1.26874316e+00 -1.09257340e-01  4.75354999e-01 -9.52289402e-01
  -4.49213207e-01  7.79702902e-01  8.99825513e-01  1.37120390e+00
   6.09281063e-01  7.85250247e-01 -6.14230216e-01 -1.58044994e-01
  -2.23495364e-01 -4.80601132e-01 -1.01770163e-02 -5.74464321e-01
   1.16622663e+00  2.97639638e-01  3.85505587e-01  6.44721270e-01
  -1.34785223e+00 -7.75588751e-02 -9.44777489e-01 -6.47249043e-01
   1.07706857e+00 -5.27694464e-01 -8.17783117e-01 -4.76593375e-02
  -1.26756716e+00 -5.07947087e-01 -6.41940057e-01 -1.10050035e+00
  -7.15758979e-01  2.76301242e-02 -6.30349219e-01 -1.78727400e+00
  -7.76470423e-01  9.76748884e-01 -8.44174504e-01 -1.23963547e+00
   9.16157067e-02 -1.25135410e+00 -3.06031704e-02  1.09661415e-01
  -4.33731735e-01  1.46347189e+00 -6.54673696e-01  9.23779368e-01
  -7.61676908e-01  5.44989435e-03 -8.16443324e-01 -2.17038083e+00
   3.81327689e-01 -6.36102796e-01 -1.63769364e+00 -6.83175147e-01
   1.06979251e+00  7.51540124e-01 -5.38114071e-01  1.10587776e+00
  -9.51599538e-01  6.52960300e-01 -8.11773121e-01 -1.58506155e-01
   1.73332006e-01  6.54552221e-01  1.14382339e+00  3.18630815e-01
   8.35734606e-02  7.91361380e+00  7.87758398e+00  7.29679441e+00
   7.32568693e+00  7.79486227e+00  7.02848625e+00  7.76348877e+00
   7.61181784e+00  7.35758829e+00  7.02959776e+00  7.81541729e+00
   7.37215424e+00  7.23868322e+00  8.31426716e+00  8.83218098e+00
   8.17538261e+00  7.65656471e+00  7.79573822e+00  7.21049023e+00
   8.03319073e+00  7.58021927e+00  7.93225145e+00  7.29150915e+00
   8.34814930e+00  7.64585209e+00  8.13688755e+00  6.72787380e+00
   7.21358109e+00  8.55158520e+00  7.00456238e+00  7.57180071e+00
   6.36446571e+00  8.18652534e+00  7.02005100e+00  7.34346676e+00
   7.59375334e+00  8.13765430e+00  7.22682333e+00  9.13684273e+00
   6.57863331e+00  6.80546999e+00  6.75992823e+00  7.18072796e+00
   7.39608383e+00  7.22835398e+00  6.83123398e+00  7.31636381e+00
   7.06261301e+00  8.85589981e+00  7.80167627e+00  8.28492451e+00
   6.65079117e+00  7.61526060e+00  7.09176779e+00  6.99132681e+00
   7.81888723e+00  6.63051796e+00  7.36835146e+00  6.70973969e+00
   2.73493052e-01  4.37674463e-01  8.29188228e-01  1.49809122e+00
   8.21185768e-01  9.67409074e-01  3.88608336e-01  2.28331017e+00
   1.77724874e+00  2.72419930e-01  4.32459474e-01  2.41576076e-01
   2.63811874e+00  1.97288489e+00  2.07742333e+00  1.67037785e-01
   1.76699138e+00  5.79720616e-01  1.80662251e+00  1.07140398e+00
   3.89254808e-01  2.54948258e-01  1.08622837e+00  3.11554813e+00
   2.33815694e+00  5.54396391e-01  1.56053758e+00  1.67389929e+00
   1.38990760e-01  7.88160563e-01  2.29844451e-01  2.74827766e+00
   1.95732558e+00  1.73988974e+00  1.51127243e+00  1.09705389e+00
   1.04623210e+00  7.35536873e-01  2.92896748e-01  2.23708212e-01
   5.23867369e-01  9.61841106e-01  1.76301610e+00  1.47582638e+00
   1.40231812e+00  5.79698324e-01  3.05175424e-01  9.67019141e-01
   1.07804990e+00  1.20919883e+00  2.12327814e+00  2.60311079e+00
   6.45673215e-01  9.75404978e-02  1.44631672e+00  4.01217282e-01
   1.43252814e+00  3.51542950e-01  1.46061599e+00  2.71110392e+00
   1.52022350e+00  5.69080114e-01  5.70945740e-01  3.82653713e-01
   5.20447969e-01  1.44529605e+00  1.42951846e-01  2.25002146e+00
   2.10991669e+00  9.25111115e-01  1.75443053e+00  1.82409120e+00
   1.96548343e+00  2.80569005e+00  1.91576397e+00  1.04599810e+00
   5.17388701e-01  6.12564981e-01  1.39354920e+00  2.68992603e-01
   3.37715745e-01  4.58869219e-01  8.04844379e-01  1.36493230e+00
   5.09543478e-01  1.90008056e+00  1.48655486e+00  2.14838433e+00
   3.74002814e-01  7.75568902e-01  1.59724963e+00  5.66096008e-01
   5.96780300e-01  3.28255355e-01  2.20429242e-01  1.99125612e+00
   1.44230235e+00  8.35174084e-01  9.11253989e-01  1.25197840e+00
   1.89989161e+00  2.05707169e+00  6.36408687e-01  2.07876253e+00
   5.33790588e-01  1.43009424e+00  1.10647357e+00  1.75541282e+00
   1.78935230e-01  6.74939811e-01  1.74507642e+00  5.28559804e-01
   7.13969469e-01  7.53190756e-01  1.85093403e-01  8.37936938e-01
   2.73394465e-01  5.28148711e-01  6.13312840e-01  6.24529362e-01
   7.46668816e+00 -6.32825804e+00 -7.21417713e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 23:20:04.015167
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.3113
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-25 23:20:04.019470
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8918.56
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-25 23:20:04.023311
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.6751
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-25 23:20:04.027194
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   -797.72
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140364576202088
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140363348917384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140363348917888
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140363349029048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140363349029552
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140363349030056

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa93efbfc50> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.556704
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.520299
grad_step = 000002, loss = 0.495034
grad_step = 000003, loss = 0.470407
grad_step = 000004, loss = 0.442535
grad_step = 000005, loss = 0.412162
grad_step = 000006, loss = 0.378584
grad_step = 000007, loss = 0.346698
grad_step = 000008, loss = 0.321713
grad_step = 000009, loss = 0.307962
grad_step = 000010, loss = 0.288755
grad_step = 000011, loss = 0.267977
grad_step = 000012, loss = 0.252829
grad_step = 000013, loss = 0.240563
grad_step = 000014, loss = 0.230949
grad_step = 000015, loss = 0.221894
grad_step = 000016, loss = 0.211906
grad_step = 000017, loss = 0.201518
grad_step = 000018, loss = 0.190788
grad_step = 000019, loss = 0.179439
grad_step = 000020, loss = 0.168627
grad_step = 000021, loss = 0.159609
grad_step = 000022, loss = 0.151780
grad_step = 000023, loss = 0.143777
grad_step = 000024, loss = 0.135456
grad_step = 000025, loss = 0.127138
grad_step = 000026, loss = 0.118981
grad_step = 000027, loss = 0.111360
grad_step = 000028, loss = 0.104559
grad_step = 000029, loss = 0.098128
grad_step = 000030, loss = 0.091661
grad_step = 000031, loss = 0.085455
grad_step = 000032, loss = 0.079987
grad_step = 000033, loss = 0.075022
grad_step = 000034, loss = 0.069981
grad_step = 000035, loss = 0.064954
grad_step = 000036, loss = 0.060233
grad_step = 000037, loss = 0.055873
grad_step = 000038, loss = 0.051926
grad_step = 000039, loss = 0.048323
grad_step = 000040, loss = 0.044815
grad_step = 000041, loss = 0.041364
grad_step = 000042, loss = 0.038189
grad_step = 000043, loss = 0.035327
grad_step = 000044, loss = 0.032626
grad_step = 000045, loss = 0.030053
grad_step = 000046, loss = 0.027681
grad_step = 000047, loss = 0.025508
grad_step = 000048, loss = 0.023468
grad_step = 000049, loss = 0.021507
grad_step = 000050, loss = 0.019648
grad_step = 000051, loss = 0.017982
grad_step = 000052, loss = 0.016522
grad_step = 000053, loss = 0.015164
grad_step = 000054, loss = 0.013866
grad_step = 000055, loss = 0.012673
grad_step = 000056, loss = 0.011603
grad_step = 000057, loss = 0.010628
grad_step = 000058, loss = 0.009708
grad_step = 000059, loss = 0.008853
grad_step = 000060, loss = 0.008107
grad_step = 000061, loss = 0.007465
grad_step = 000062, loss = 0.006873
grad_step = 000063, loss = 0.006309
grad_step = 000064, loss = 0.005796
grad_step = 000065, loss = 0.005360
grad_step = 000066, loss = 0.004980
grad_step = 000067, loss = 0.004629
grad_step = 000068, loss = 0.004318
grad_step = 000069, loss = 0.004054
grad_step = 000070, loss = 0.003818
grad_step = 000071, loss = 0.003604
grad_step = 000072, loss = 0.003414
grad_step = 000073, loss = 0.003252
grad_step = 000074, loss = 0.003122
grad_step = 000075, loss = 0.003003
grad_step = 000076, loss = 0.002897
grad_step = 000077, loss = 0.002815
grad_step = 000078, loss = 0.002743
grad_step = 000079, loss = 0.002680
grad_step = 000080, loss = 0.002626
grad_step = 000081, loss = 0.002579
grad_step = 000082, loss = 0.002543
grad_step = 000083, loss = 0.002509
grad_step = 000084, loss = 0.002481
grad_step = 000085, loss = 0.002462
grad_step = 000086, loss = 0.002444
grad_step = 000087, loss = 0.002425
grad_step = 000088, loss = 0.002408
grad_step = 000089, loss = 0.002395
grad_step = 000090, loss = 0.002385
grad_step = 000091, loss = 0.002373
grad_step = 000092, loss = 0.002363
grad_step = 000093, loss = 0.002354
grad_step = 000094, loss = 0.002344
grad_step = 000095, loss = 0.002335
grad_step = 000096, loss = 0.002327
grad_step = 000097, loss = 0.002320
grad_step = 000098, loss = 0.002313
grad_step = 000099, loss = 0.002305
grad_step = 000100, loss = 0.002297
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002290
grad_step = 000102, loss = 0.002284
grad_step = 000103, loss = 0.002277
grad_step = 000104, loss = 0.002271
grad_step = 000105, loss = 0.002265
grad_step = 000106, loss = 0.002259
grad_step = 000107, loss = 0.002253
grad_step = 000108, loss = 0.002248
grad_step = 000109, loss = 0.002243
grad_step = 000110, loss = 0.002238
grad_step = 000111, loss = 0.002233
grad_step = 000112, loss = 0.002228
grad_step = 000113, loss = 0.002224
grad_step = 000114, loss = 0.002220
grad_step = 000115, loss = 0.002216
grad_step = 000116, loss = 0.002212
grad_step = 000117, loss = 0.002209
grad_step = 000118, loss = 0.002205
grad_step = 000119, loss = 0.002202
grad_step = 000120, loss = 0.002199
grad_step = 000121, loss = 0.002196
grad_step = 000122, loss = 0.002193
grad_step = 000123, loss = 0.002190
grad_step = 000124, loss = 0.002187
grad_step = 000125, loss = 0.002184
grad_step = 000126, loss = 0.002181
grad_step = 000127, loss = 0.002178
grad_step = 000128, loss = 0.002176
grad_step = 000129, loss = 0.002173
grad_step = 000130, loss = 0.002170
grad_step = 000131, loss = 0.002168
grad_step = 000132, loss = 0.002165
grad_step = 000133, loss = 0.002162
grad_step = 000134, loss = 0.002160
grad_step = 000135, loss = 0.002157
grad_step = 000136, loss = 0.002154
grad_step = 000137, loss = 0.002152
grad_step = 000138, loss = 0.002149
grad_step = 000139, loss = 0.002146
grad_step = 000140, loss = 0.002144
grad_step = 000141, loss = 0.002141
grad_step = 000142, loss = 0.002138
grad_step = 000143, loss = 0.002135
grad_step = 000144, loss = 0.002132
grad_step = 000145, loss = 0.002130
grad_step = 000146, loss = 0.002127
grad_step = 000147, loss = 0.002124
grad_step = 000148, loss = 0.002121
grad_step = 000149, loss = 0.002118
grad_step = 000150, loss = 0.002115
grad_step = 000151, loss = 0.002112
grad_step = 000152, loss = 0.002109
grad_step = 000153, loss = 0.002106
grad_step = 000154, loss = 0.002103
grad_step = 000155, loss = 0.002099
grad_step = 000156, loss = 0.002096
grad_step = 000157, loss = 0.002093
grad_step = 000158, loss = 0.002089
grad_step = 000159, loss = 0.002086
grad_step = 000160, loss = 0.002082
grad_step = 000161, loss = 0.002079
grad_step = 000162, loss = 0.002075
grad_step = 000163, loss = 0.002071
grad_step = 000164, loss = 0.002067
grad_step = 000165, loss = 0.002063
grad_step = 000166, loss = 0.002059
grad_step = 000167, loss = 0.002055
grad_step = 000168, loss = 0.002051
grad_step = 000169, loss = 0.002048
grad_step = 000170, loss = 0.002043
grad_step = 000171, loss = 0.002038
grad_step = 000172, loss = 0.002034
grad_step = 000173, loss = 0.002030
grad_step = 000174, loss = 0.002025
grad_step = 000175, loss = 0.002021
grad_step = 000176, loss = 0.002017
grad_step = 000177, loss = 0.002013
grad_step = 000178, loss = 0.002008
grad_step = 000179, loss = 0.002003
grad_step = 000180, loss = 0.001998
grad_step = 000181, loss = 0.001993
grad_step = 000182, loss = 0.001989
grad_step = 000183, loss = 0.001984
grad_step = 000184, loss = 0.001979
grad_step = 000185, loss = 0.001974
grad_step = 000186, loss = 0.001969
grad_step = 000187, loss = 0.001964
grad_step = 000188, loss = 0.001959
grad_step = 000189, loss = 0.001955
grad_step = 000190, loss = 0.001952
grad_step = 000191, loss = 0.001949
grad_step = 000192, loss = 0.001943
grad_step = 000193, loss = 0.001936
grad_step = 000194, loss = 0.001929
grad_step = 000195, loss = 0.001922
grad_step = 000196, loss = 0.001918
grad_step = 000197, loss = 0.001914
grad_step = 000198, loss = 0.001910
grad_step = 000199, loss = 0.001906
grad_step = 000200, loss = 0.001902
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001898
grad_step = 000202, loss = 0.001892
grad_step = 000203, loss = 0.001885
grad_step = 000204, loss = 0.001878
grad_step = 000205, loss = 0.001871
grad_step = 000206, loss = 0.001865
grad_step = 000207, loss = 0.001858
grad_step = 000208, loss = 0.001853
grad_step = 000209, loss = 0.001848
grad_step = 000210, loss = 0.001843
grad_step = 000211, loss = 0.001841
grad_step = 000212, loss = 0.001846
grad_step = 000213, loss = 0.001876
grad_step = 000214, loss = 0.001926
grad_step = 000215, loss = 0.001960
grad_step = 000216, loss = 0.001860
grad_step = 000217, loss = 0.001806
grad_step = 000218, loss = 0.001858
grad_step = 000219, loss = 0.001878
grad_step = 000220, loss = 0.001837
grad_step = 000221, loss = 0.001793
grad_step = 000222, loss = 0.001862
grad_step = 000223, loss = 0.001923
grad_step = 000224, loss = 0.001799
grad_step = 000225, loss = 0.001834
grad_step = 000226, loss = 0.001939
grad_step = 000227, loss = 0.001871
grad_step = 000228, loss = 0.001787
grad_step = 000229, loss = 0.001932
grad_step = 000230, loss = 0.001895
grad_step = 000231, loss = 0.001751
grad_step = 000232, loss = 0.001873
grad_step = 000233, loss = 0.001864
grad_step = 000234, loss = 0.001855
grad_step = 000235, loss = 0.001751
grad_step = 000236, loss = 0.001934
grad_step = 000237, loss = 0.001910
grad_step = 000238, loss = 0.001791
grad_step = 000239, loss = 0.001876
grad_step = 000240, loss = 0.001822
grad_step = 000241, loss = 0.001810
grad_step = 000242, loss = 0.001751
grad_step = 000243, loss = 0.001828
grad_step = 000244, loss = 0.001698
grad_step = 000245, loss = 0.001781
grad_step = 000246, loss = 0.001724
grad_step = 000247, loss = 0.001724
grad_step = 000248, loss = 0.001741
grad_step = 000249, loss = 0.001682
grad_step = 000250, loss = 0.001742
grad_step = 000251, loss = 0.001675
grad_step = 000252, loss = 0.001706
grad_step = 000253, loss = 0.001684
grad_step = 000254, loss = 0.001673
grad_step = 000255, loss = 0.001693
grad_step = 000256, loss = 0.001655
grad_step = 000257, loss = 0.001676
grad_step = 000258, loss = 0.001658
grad_step = 000259, loss = 0.001653
grad_step = 000260, loss = 0.001664
grad_step = 000261, loss = 0.001639
grad_step = 000262, loss = 0.001649
grad_step = 000263, loss = 0.001640
grad_step = 000264, loss = 0.001632
grad_step = 000265, loss = 0.001640
grad_step = 000266, loss = 0.001625
grad_step = 000267, loss = 0.001628
grad_step = 000268, loss = 0.001626
grad_step = 000269, loss = 0.001615
grad_step = 000270, loss = 0.001621
grad_step = 000271, loss = 0.001613
grad_step = 000272, loss = 0.001609
grad_step = 000273, loss = 0.001612
grad_step = 000274, loss = 0.001604
grad_step = 000275, loss = 0.001603
grad_step = 000276, loss = 0.001603
grad_step = 000277, loss = 0.001596
grad_step = 000278, loss = 0.001596
grad_step = 000279, loss = 0.001594
grad_step = 000280, loss = 0.001590
grad_step = 000281, loss = 0.001591
grad_step = 000282, loss = 0.001592
grad_step = 000283, loss = 0.001593
grad_step = 000284, loss = 0.001600
grad_step = 000285, loss = 0.001612
grad_step = 000286, loss = 0.001625
grad_step = 000287, loss = 0.001648
grad_step = 000288, loss = 0.001659
grad_step = 000289, loss = 0.001659
grad_step = 000290, loss = 0.001638
grad_step = 000291, loss = 0.001598
grad_step = 000292, loss = 0.001565
grad_step = 000293, loss = 0.001553
grad_step = 000294, loss = 0.001564
grad_step = 000295, loss = 0.001584
grad_step = 000296, loss = 0.001591
grad_step = 000297, loss = 0.001580
grad_step = 000298, loss = 0.001556
grad_step = 000299, loss = 0.001538
grad_step = 000300, loss = 0.001535
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001543
grad_step = 000302, loss = 0.001552
grad_step = 000303, loss = 0.001551
grad_step = 000304, loss = 0.001541
grad_step = 000305, loss = 0.001527
grad_step = 000306, loss = 0.001517
grad_step = 000307, loss = 0.001513
grad_step = 000308, loss = 0.001516
grad_step = 000309, loss = 0.001520
grad_step = 000310, loss = 0.001521
grad_step = 000311, loss = 0.001519
grad_step = 000312, loss = 0.001511
grad_step = 000313, loss = 0.001502
grad_step = 000314, loss = 0.001495
grad_step = 000315, loss = 0.001492
grad_step = 000316, loss = 0.001491
grad_step = 000317, loss = 0.001492
grad_step = 000318, loss = 0.001494
grad_step = 000319, loss = 0.001497
grad_step = 000320, loss = 0.001498
grad_step = 000321, loss = 0.001504
grad_step = 000322, loss = 0.001511
grad_step = 000323, loss = 0.001539
grad_step = 000324, loss = 0.001543
grad_step = 000325, loss = 0.001561
grad_step = 000326, loss = 0.001511
grad_step = 000327, loss = 0.001474
grad_step = 000328, loss = 0.001467
grad_step = 000329, loss = 0.001488
grad_step = 000330, loss = 0.001513
grad_step = 000331, loss = 0.001494
grad_step = 000332, loss = 0.001469
grad_step = 000333, loss = 0.001446
grad_step = 000334, loss = 0.001444
grad_step = 000335, loss = 0.001459
grad_step = 000336, loss = 0.001466
grad_step = 000337, loss = 0.001465
grad_step = 000338, loss = 0.001448
grad_step = 000339, loss = 0.001444
grad_step = 000340, loss = 0.001473
grad_step = 000341, loss = 0.001538
grad_step = 000342, loss = 0.001619
grad_step = 000343, loss = 0.001658
grad_step = 000344, loss = 0.001596
grad_step = 000345, loss = 0.001486
grad_step = 000346, loss = 0.001431
grad_step = 000347, loss = 0.001481
grad_step = 000348, loss = 0.001532
grad_step = 000349, loss = 0.001507
grad_step = 000350, loss = 0.001431
grad_step = 000351, loss = 0.001425
grad_step = 000352, loss = 0.001475
grad_step = 000353, loss = 0.001475
grad_step = 000354, loss = 0.001431
grad_step = 000355, loss = 0.001413
grad_step = 000356, loss = 0.001435
grad_step = 000357, loss = 0.001448
grad_step = 000358, loss = 0.001428
grad_step = 000359, loss = 0.001411
grad_step = 000360, loss = 0.001412
grad_step = 000361, loss = 0.001421
grad_step = 000362, loss = 0.001420
grad_step = 000363, loss = 0.001406
grad_step = 000364, loss = 0.001397
grad_step = 000365, loss = 0.001404
grad_step = 000366, loss = 0.001409
grad_step = 000367, loss = 0.001402
grad_step = 000368, loss = 0.001392
grad_step = 000369, loss = 0.001391
grad_step = 000370, loss = 0.001395
grad_step = 000371, loss = 0.001395
grad_step = 000372, loss = 0.001389
grad_step = 000373, loss = 0.001384
grad_step = 000374, loss = 0.001383
grad_step = 000375, loss = 0.001385
grad_step = 000376, loss = 0.001385
grad_step = 000377, loss = 0.001382
grad_step = 000378, loss = 0.001378
grad_step = 000379, loss = 0.001376
grad_step = 000380, loss = 0.001378
grad_step = 000381, loss = 0.001382
grad_step = 000382, loss = 0.001385
grad_step = 000383, loss = 0.001395
grad_step = 000384, loss = 0.001410
grad_step = 000385, loss = 0.001451
grad_step = 000386, loss = 0.001473
grad_step = 000387, loss = 0.001504
grad_step = 000388, loss = 0.001437
grad_step = 000389, loss = 0.001384
grad_step = 000390, loss = 0.001367
grad_step = 000391, loss = 0.001391
grad_step = 000392, loss = 0.001415
grad_step = 000393, loss = 0.001389
grad_step = 000394, loss = 0.001364
grad_step = 000395, loss = 0.001363
grad_step = 000396, loss = 0.001381
grad_step = 000397, loss = 0.001393
grad_step = 000398, loss = 0.001372
grad_step = 000399, loss = 0.001355
grad_step = 000400, loss = 0.001354
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001363
grad_step = 000402, loss = 0.001370
grad_step = 000403, loss = 0.001362
grad_step = 000404, loss = 0.001350
grad_step = 000405, loss = 0.001344
grad_step = 000406, loss = 0.001348
grad_step = 000407, loss = 0.001354
grad_step = 000408, loss = 0.001351
grad_step = 000409, loss = 0.001343
grad_step = 000410, loss = 0.001337
grad_step = 000411, loss = 0.001336
grad_step = 000412, loss = 0.001338
grad_step = 000413, loss = 0.001340
grad_step = 000414, loss = 0.001339
grad_step = 000415, loss = 0.001336
grad_step = 000416, loss = 0.001335
grad_step = 000417, loss = 0.001337
grad_step = 000418, loss = 0.001345
grad_step = 000419, loss = 0.001359
grad_step = 000420, loss = 0.001389
grad_step = 000421, loss = 0.001397
grad_step = 000422, loss = 0.001422
grad_step = 000423, loss = 0.001399
grad_step = 000424, loss = 0.001392
grad_step = 000425, loss = 0.001359
grad_step = 000426, loss = 0.001341
grad_step = 000427, loss = 0.001328
grad_step = 000428, loss = 0.001327
grad_step = 000429, loss = 0.001341
grad_step = 000430, loss = 0.001348
grad_step = 000431, loss = 0.001345
grad_step = 000432, loss = 0.001322
grad_step = 000433, loss = 0.001309
grad_step = 000434, loss = 0.001312
grad_step = 000435, loss = 0.001324
grad_step = 000436, loss = 0.001333
grad_step = 000437, loss = 0.001326
grad_step = 000438, loss = 0.001315
grad_step = 000439, loss = 0.001307
grad_step = 000440, loss = 0.001309
grad_step = 000441, loss = 0.001312
grad_step = 000442, loss = 0.001315
grad_step = 000443, loss = 0.001306
grad_step = 000444, loss = 0.001296
grad_step = 000445, loss = 0.001293
grad_step = 000446, loss = 0.001298
grad_step = 000447, loss = 0.001300
grad_step = 000448, loss = 0.001295
grad_step = 000449, loss = 0.001289
grad_step = 000450, loss = 0.001287
grad_step = 000451, loss = 0.001289
grad_step = 000452, loss = 0.001291
grad_step = 000453, loss = 0.001288
grad_step = 000454, loss = 0.001284
grad_step = 000455, loss = 0.001281
grad_step = 000456, loss = 0.001279
grad_step = 000457, loss = 0.001280
grad_step = 000458, loss = 0.001281
grad_step = 000459, loss = 0.001282
grad_step = 000460, loss = 0.001283
grad_step = 000461, loss = 0.001291
grad_step = 000462, loss = 0.001311
grad_step = 000463, loss = 0.001372
grad_step = 000464, loss = 0.001443
grad_step = 000465, loss = 0.001580
grad_step = 000466, loss = 0.001490
grad_step = 000467, loss = 0.001404
grad_step = 000468, loss = 0.001312
grad_step = 000469, loss = 0.001341
grad_step = 000470, loss = 0.001376
grad_step = 000471, loss = 0.001323
grad_step = 000472, loss = 0.001301
grad_step = 000473, loss = 0.001326
grad_step = 000474, loss = 0.001325
grad_step = 000475, loss = 0.001305
grad_step = 000476, loss = 0.001284
grad_step = 000477, loss = 0.001300
grad_step = 000478, loss = 0.001313
grad_step = 000479, loss = 0.001296
grad_step = 000480, loss = 0.001263
grad_step = 000481, loss = 0.001265
grad_step = 000482, loss = 0.001288
grad_step = 000483, loss = 0.001285
grad_step = 000484, loss = 0.001260
grad_step = 000485, loss = 0.001252
grad_step = 000486, loss = 0.001261
grad_step = 000487, loss = 0.001267
grad_step = 000488, loss = 0.001259
grad_step = 000489, loss = 0.001247
grad_step = 000490, loss = 0.001246
grad_step = 000491, loss = 0.001252
grad_step = 000492, loss = 0.001254
grad_step = 000493, loss = 0.001246
grad_step = 000494, loss = 0.001241
grad_step = 000495, loss = 0.001241
grad_step = 000496, loss = 0.001243
grad_step = 000497, loss = 0.001242
grad_step = 000498, loss = 0.001240
grad_step = 000499, loss = 0.001235
grad_step = 000500, loss = 0.001235
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001238
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

  date_run                              2020-05-25 23:20:26.505296
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.246132
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-25 23:20:26.511847
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.192236
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-25 23:20:26.519207
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.129832
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-25 23:20:26.524814
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.92109
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
100%|██████████| 10/10 [00:02<00:00,  3.53it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 2.837 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.240294
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.240293645858765 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1bc8c50> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  7.20it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.390 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1f2e198> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:12<00:29,  4.15s/it, avg_epoch_loss=6.94] 60%|██████    | 6/10 [00:23<00:15,  3.97s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:33<00:03,  3.85s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:37<00:00,  3.74s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 37.365 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.861563
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.861563205718994 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1fb83c8> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.56it/s, avg_epoch_loss=5.83]
INFO:root:Epoch[0] Elapsed time 1.800 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.834928
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.834927701950074 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1edc358> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [02:04<18:41, 124.59s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [04:45<18:04, 135.56s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [07:37<17:04, 146.34s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [10:36<15:36, 156.09s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [13:29<13:26, 161.27s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [16:30<11:09, 167.29s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [19:28<08:31, 170.35s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [22:25<05:44, 172.43s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [25:23<02:54, 174.03s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [28:17<00:00, 174.21s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [28:18<00:00, 169.80s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 1698.022 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1c65320> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  7.57it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 1.341 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1bb0d30> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 38.88it/s, avg_epoch_loss=5.16]
INFO:root:Epoch[0] Elapsed time 0.258 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.159531
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.159530782699585 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa8e1c1af60> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
0   2020-05-25 23:19:54.704345  ...    mean_absolute_error
1   2020-05-25 23:19:54.708640  ...     mean_squared_error
2   2020-05-25 23:19:54.712262  ...  median_absolute_error
3   2020-05-25 23:19:54.715798  ...               r2_score
4   2020-05-25 23:20:04.015167  ...    mean_absolute_error
5   2020-05-25 23:20:04.019470  ...     mean_squared_error
6   2020-05-25 23:20:04.023311  ...  median_absolute_error
7   2020-05-25 23:20:04.027194  ...               r2_score
8   2020-05-25 23:20:26.505296  ...    mean_absolute_error
9   2020-05-25 23:20:26.511847  ...     mean_squared_error
10  2020-05-25 23:20:26.519207  ...  median_absolute_error
11  2020-05-25 23:20:26.524814  ...               r2_score

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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fae30ffe9e8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355447.3125
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 266818.9375
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 180637.7969
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 109401.1094
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 63649.0938
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 37824.9805
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 23591.8594
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 15474.5645
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 10680.0889
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 7727.3120

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.2001508   1.9570178   3.1424572   2.5986903   1.6159432   2.68553
   3.3453643   2.2362967   2.4067588   1.7536963   1.8791883   2.4968903
   2.669754    0.6287121   1.811993    2.9511607   1.9249382   2.8003242
   3.8964942   0.66398895  1.4416856   0.41706544  2.6520786   0.87216556
   2.7417605   1.9624082   2.6524339   0.40504515  2.3294857   3.755403
   3.316142    2.6293015   2.5672095   1.6015325   1.475251    2.4262698
   3.5810592   0.5984271   3.7369344   2.6437511   3.874663    3.1105027
   3.937055    2.3189957   2.6964734   1.7878656   3.3599794   1.0563822
   1.8536047   3.5373876   2.6851904   1.7601848   1.9015437   2.2805371
   1.3899517   1.489444    2.993182    2.9796154   2.6310008   2.324724
  -0.26178715  9.353587    6.145688    7.594642    5.477923    8.318471
   7.625364    6.7360973   7.6910152   5.859969    6.9867263   7.6549664
   9.280575    7.1124043   7.4018216   8.286108    7.633419    8.404147
   9.055973    8.486421    7.6629405   6.9065347   7.1163497   5.8349233
   7.4340343   6.9219346   9.182837    7.6494446   7.704532    8.286095
   7.8132873   8.176736    7.263225    7.8223166   8.291272    7.672552
   7.435613    7.57343     6.320163    6.634929    5.898044    7.669543
   7.201299    6.114375    7.0487757   6.786587    5.961644    6.411976
   7.306807    7.8591423   8.75911     7.461471    8.452717    7.351503
   8.637522    7.687986    7.521033    8.305629    6.8767023   6.3902698
  -1.1147547  -2.0296235  -0.10875505 -1.0143806   0.85653424 -0.18652256
  -0.33689934  0.07505056  0.52773905  0.3664744   0.7406107   1.1142719
  -1.4248316  -0.68355745  1.068458   -0.5321584   0.3920786   0.5841378
  -0.7014593  -0.4582693  -1.0956855   1.2560394  -0.97039425 -1.4382169
  -0.02717167  1.0374264   0.5726442   0.46295166 -0.07508849  0.5048327
  -0.47985053 -0.12979925 -1.2729466   0.916075    1.2443259  -1.4542484
   0.44269854 -0.6589302   1.589575    0.12050873 -0.49263903 -0.50002193
  -1.5839291  -1.4052241   0.7701299   0.05962527  0.54165107 -0.48187473
  -0.2887855  -0.28461084  0.51496613  0.44436866  0.3612229  -0.589221
  -0.19846493 -1.2904626   0.20188269  1.2430016  -1.1728914  -0.01709584
   0.10280621  3.6726446   3.0146022   1.9461497   1.8972142   2.8710887
   3.2886038   2.5129356   3.9344769   3.3684344   4.1382723   3.7252498
   3.153006    2.036856    3.3283882   3.4740252   2.0112119   4.0696564
   5.1492624   4.0461864   4.7374487   2.7295551   2.1955466   3.0441647
   4.040676    2.9392242   3.0276523   5.4198174   5.630363    3.100513
   2.9278598   3.6479964   3.7524037   2.1313117   2.6965966   3.8072958
   3.013205    3.2724056   2.5689635   3.4467182   3.3670478   3.215991
   4.0645804   4.5805187   4.853898    3.837882    3.7104173   4.2147865
   2.8156428   4.9721026   2.786896    3.982768    3.1397605   4.551263
   2.981956    5.2645845   4.0205526   3.8505082   3.9646688   1.9407012
   0.1031031   9.6031475   7.000309    7.4965205   8.676715    7.793342
   7.855792    7.0314894   7.931428    6.7771783   7.744169    7.881583
   7.2302456   7.5119967   6.8160133   8.388518    9.909299    7.6721396
   7.855301    6.4545455   8.447994    8.277454    9.676249    8.696967
   8.179631    8.868815    9.410223    8.351894    9.318228    6.9407964
   8.346635    8.113241    7.695943    6.868685    7.7335835   9.744912
   9.060788    7.675575    7.478553    8.874549    6.5423207   7.8982654
   8.475375    7.3878894   6.8388586   8.362974    7.542526    6.922314
   7.597962    7.137532    8.126786    8.309462    8.291766    7.5334806
   8.212679    8.664123    7.1571193   7.7852387   7.516442    8.208283
   0.51322454  0.60076994  0.6932176   2.1560779   1.2490177   1.0135584
   1.1162095   0.6981201   1.6327565   0.22805685  2.1911092   0.709756
   2.3670921   2.3995357   0.27678144  1.1934751   0.92609406  1.0377098
   1.3297029   1.9561298   0.49409074  1.028308    1.2339382   0.4294963
   0.09455681  0.28079128  0.21146822  0.7429472   0.7319597   0.9087072
   0.2267003   0.6228261   0.621002    0.38537467  0.51912796  0.84021926
   1.7754552   0.4462334   2.044631    2.1873782   0.6116891   1.0708696
   1.3888975   0.34969997  1.4737198   0.1200074   0.94617164  0.30286252
   0.70923257  1.4276774   1.5249233   1.2855811   0.5811416   2.4192736
   2.4087963   2.843135    0.51301104  0.20040119  0.59753484  0.47754514
  -8.142208    8.091419   -6.1119413 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 23:50:00.767821
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.6235
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-25 23:50:00.772687
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8974.13
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-25 23:50:00.776421
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   93.9993
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-25 23:50:00.779872
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -802.696
metric_name                                             r2_score
Name: 3, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140385609829288
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328685192
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328685696
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328288952
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328289456
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140383328289960

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fae45343438> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.587050
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.561916
grad_step = 000002, loss = 0.541793
grad_step = 000003, loss = 0.520126
grad_step = 000004, loss = 0.498513
grad_step = 000005, loss = 0.480177
grad_step = 000006, loss = 0.461525
grad_step = 000007, loss = 0.449246
grad_step = 000008, loss = 0.437747
grad_step = 000009, loss = 0.420483
grad_step = 000010, loss = 0.405302
grad_step = 000011, loss = 0.394317
grad_step = 000012, loss = 0.385426
grad_step = 000013, loss = 0.375614
grad_step = 000014, loss = 0.364080
grad_step = 000015, loss = 0.351789
grad_step = 000016, loss = 0.339817
grad_step = 000017, loss = 0.328966
grad_step = 000018, loss = 0.318169
grad_step = 000019, loss = 0.306596
grad_step = 000020, loss = 0.294172
grad_step = 000021, loss = 0.282086
grad_step = 000022, loss = 0.271267
grad_step = 000023, loss = 0.261464
grad_step = 000024, loss = 0.251714
grad_step = 000025, loss = 0.241488
grad_step = 000026, loss = 0.231311
grad_step = 000027, loss = 0.221924
grad_step = 000028, loss = 0.213180
grad_step = 000029, loss = 0.204352
grad_step = 000030, loss = 0.195210
grad_step = 000031, loss = 0.186251
grad_step = 000032, loss = 0.177978
grad_step = 000033, loss = 0.169987
grad_step = 000034, loss = 0.161910
grad_step = 000035, loss = 0.153957
grad_step = 000036, loss = 0.146228
grad_step = 000037, loss = 0.139027
grad_step = 000038, loss = 0.132073
grad_step = 000039, loss = 0.125203
grad_step = 000040, loss = 0.118656
grad_step = 000041, loss = 0.112483
grad_step = 000042, loss = 0.106463
grad_step = 000043, loss = 0.100513
grad_step = 000044, loss = 0.094787
grad_step = 000045, loss = 0.089434
grad_step = 000046, loss = 0.084300
grad_step = 000047, loss = 0.077838
grad_step = 000048, loss = 0.071586
grad_step = 000049, loss = 0.066198
grad_step = 000050, loss = 0.061874
grad_step = 000051, loss = 0.058092
grad_step = 000052, loss = 0.054448
grad_step = 000053, loss = 0.050751
grad_step = 000054, loss = 0.047056
grad_step = 000055, loss = 0.043377
grad_step = 000056, loss = 0.039942
grad_step = 000057, loss = 0.036720
grad_step = 000058, loss = 0.033768
grad_step = 000059, loss = 0.031137
grad_step = 000060, loss = 0.028758
grad_step = 000061, loss = 0.026568
grad_step = 000062, loss = 0.024454
grad_step = 000063, loss = 0.022416
grad_step = 000064, loss = 0.020453
grad_step = 000065, loss = 0.018620
grad_step = 000066, loss = 0.016574
grad_step = 000067, loss = 0.014980
grad_step = 000068, loss = 0.013861
grad_step = 000069, loss = 0.012813
grad_step = 000070, loss = 0.011740
grad_step = 000071, loss = 0.010700
grad_step = 000072, loss = 0.009776
grad_step = 000073, loss = 0.009012
grad_step = 000074, loss = 0.008275
grad_step = 000075, loss = 0.007559
grad_step = 000076, loss = 0.006949
grad_step = 000077, loss = 0.006420
grad_step = 000078, loss = 0.005987
grad_step = 000079, loss = 0.005591
grad_step = 000080, loss = 0.005203
grad_step = 000081, loss = 0.004854
grad_step = 000082, loss = 0.004519
grad_step = 000083, loss = 0.004210
grad_step = 000084, loss = 0.003958
grad_step = 000085, loss = 0.003743
grad_step = 000086, loss = 0.003555
grad_step = 000087, loss = 0.003365
grad_step = 000088, loss = 0.003190
grad_step = 000089, loss = 0.003056
grad_step = 000090, loss = 0.002941
grad_step = 000091, loss = 0.002832
grad_step = 000092, loss = 0.002727
grad_step = 000093, loss = 0.002632
grad_step = 000094, loss = 0.002565
grad_step = 000095, loss = 0.002507
grad_step = 000096, loss = 0.002447
grad_step = 000097, loss = 0.002388
grad_step = 000098, loss = 0.002335
grad_step = 000099, loss = 0.002298
grad_step = 000100, loss = 0.002268
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002236
grad_step = 000102, loss = 0.002208
grad_step = 000103, loss = 0.002186
grad_step = 000104, loss = 0.002175
grad_step = 000105, loss = 0.002169
grad_step = 000106, loss = 0.002165
grad_step = 000107, loss = 0.002155
grad_step = 000108, loss = 0.002140
grad_step = 000109, loss = 0.002118
grad_step = 000110, loss = 0.002097
grad_step = 000111, loss = 0.002084
grad_step = 000112, loss = 0.002081
grad_step = 000113, loss = 0.002083
grad_step = 000114, loss = 0.002087
grad_step = 000115, loss = 0.002091
grad_step = 000116, loss = 0.002091
grad_step = 000117, loss = 0.002089
grad_step = 000118, loss = 0.002079
grad_step = 000119, loss = 0.002066
grad_step = 000120, loss = 0.002049
grad_step = 000121, loss = 0.002035
grad_step = 000122, loss = 0.002024
grad_step = 000123, loss = 0.002016
grad_step = 000124, loss = 0.002010
grad_step = 000125, loss = 0.002007
grad_step = 000126, loss = 0.002007
grad_step = 000127, loss = 0.002015
grad_step = 000128, loss = 0.002049
grad_step = 000129, loss = 0.002121
grad_step = 000130, loss = 0.002252
grad_step = 000131, loss = 0.002224
grad_step = 000132, loss = 0.002124
grad_step = 000133, loss = 0.002013
grad_step = 000134, loss = 0.002023
grad_step = 000135, loss = 0.002111
grad_step = 000136, loss = 0.002068
grad_step = 000137, loss = 0.001957
grad_step = 000138, loss = 0.002020
grad_step = 000139, loss = 0.002066
grad_step = 000140, loss = 0.001973
grad_step = 000141, loss = 0.001954
grad_step = 000142, loss = 0.002011
grad_step = 000143, loss = 0.001988
grad_step = 000144, loss = 0.001964
grad_step = 000145, loss = 0.001974
grad_step = 000146, loss = 0.001933
grad_step = 000147, loss = 0.001960
grad_step = 000148, loss = 0.001978
grad_step = 000149, loss = 0.001921
grad_step = 000150, loss = 0.001923
grad_step = 000151, loss = 0.001954
grad_step = 000152, loss = 0.001927
grad_step = 000153, loss = 0.001907
grad_step = 000154, loss = 0.001923
grad_step = 000155, loss = 0.001906
grad_step = 000156, loss = 0.001910
grad_step = 000157, loss = 0.001913
grad_step = 000158, loss = 0.001888
grad_step = 000159, loss = 0.001887
grad_step = 000160, loss = 0.001900
grad_step = 000161, loss = 0.001896
grad_step = 000162, loss = 0.001880
grad_step = 000163, loss = 0.001884
grad_step = 000164, loss = 0.001877
grad_step = 000165, loss = 0.001865
grad_step = 000166, loss = 0.001868
grad_step = 000167, loss = 0.001871
grad_step = 000168, loss = 0.001869
grad_step = 000169, loss = 0.001863
grad_step = 000170, loss = 0.001866
grad_step = 000171, loss = 0.001864
grad_step = 000172, loss = 0.001853
grad_step = 000173, loss = 0.001848
grad_step = 000174, loss = 0.001845
grad_step = 000175, loss = 0.001842
grad_step = 000176, loss = 0.001837
grad_step = 000177, loss = 0.001834
grad_step = 000178, loss = 0.001835
grad_step = 000179, loss = 0.001835
grad_step = 000180, loss = 0.001836
grad_step = 000181, loss = 0.001844
grad_step = 000182, loss = 0.001862
grad_step = 000183, loss = 0.001909
grad_step = 000184, loss = 0.001953
grad_step = 000185, loss = 0.002003
grad_step = 000186, loss = 0.001931
grad_step = 000187, loss = 0.001842
grad_step = 000188, loss = 0.001818
grad_step = 000189, loss = 0.001866
grad_step = 000190, loss = 0.001887
grad_step = 000191, loss = 0.001837
grad_step = 000192, loss = 0.001807
grad_step = 000193, loss = 0.001828
grad_step = 000194, loss = 0.001847
grad_step = 000195, loss = 0.001837
grad_step = 000196, loss = 0.001800
grad_step = 000197, loss = 0.001786
grad_step = 000198, loss = 0.001808
grad_step = 000199, loss = 0.001820
grad_step = 000200, loss = 0.001802
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001780
grad_step = 000202, loss = 0.001782
grad_step = 000203, loss = 0.001792
grad_step = 000204, loss = 0.001790
grad_step = 000205, loss = 0.001780
grad_step = 000206, loss = 0.001768
grad_step = 000207, loss = 0.001768
grad_step = 000208, loss = 0.001777
grad_step = 000209, loss = 0.001778
grad_step = 000210, loss = 0.001772
grad_step = 000211, loss = 0.001764
grad_step = 000212, loss = 0.001758
grad_step = 000213, loss = 0.001756
grad_step = 000214, loss = 0.001759
grad_step = 000215, loss = 0.001764
grad_step = 000216, loss = 0.001763
grad_step = 000217, loss = 0.001763
grad_step = 000218, loss = 0.001761
grad_step = 000219, loss = 0.001764
grad_step = 000220, loss = 0.001767
grad_step = 000221, loss = 0.001783
grad_step = 000222, loss = 0.001787
grad_step = 000223, loss = 0.001785
grad_step = 000224, loss = 0.001755
grad_step = 000225, loss = 0.001732
grad_step = 000226, loss = 0.001730
grad_step = 000227, loss = 0.001745
grad_step = 000228, loss = 0.001768
grad_step = 000229, loss = 0.001773
grad_step = 000230, loss = 0.001772
grad_step = 000231, loss = 0.001753
grad_step = 000232, loss = 0.001747
grad_step = 000233, loss = 0.001761
grad_step = 000234, loss = 0.001810
grad_step = 000235, loss = 0.001862
grad_step = 000236, loss = 0.001931
grad_step = 000237, loss = 0.001897
grad_step = 000238, loss = 0.001833
grad_step = 000239, loss = 0.001737
grad_step = 000240, loss = 0.001728
grad_step = 000241, loss = 0.001790
grad_step = 000242, loss = 0.001824
grad_step = 000243, loss = 0.001786
grad_step = 000244, loss = 0.001716
grad_step = 000245, loss = 0.001703
grad_step = 000246, loss = 0.001742
grad_step = 000247, loss = 0.001755
grad_step = 000248, loss = 0.001728
grad_step = 000249, loss = 0.001696
grad_step = 000250, loss = 0.001705
grad_step = 000251, loss = 0.001731
grad_step = 000252, loss = 0.001721
grad_step = 000253, loss = 0.001693
grad_step = 000254, loss = 0.001681
grad_step = 000255, loss = 0.001693
grad_step = 000256, loss = 0.001701
grad_step = 000257, loss = 0.001689
grad_step = 000258, loss = 0.001672
grad_step = 000259, loss = 0.001669
grad_step = 000260, loss = 0.001677
grad_step = 000261, loss = 0.001681
grad_step = 000262, loss = 0.001672
grad_step = 000263, loss = 0.001662
grad_step = 000264, loss = 0.001659
grad_step = 000265, loss = 0.001665
grad_step = 000266, loss = 0.001671
grad_step = 000267, loss = 0.001678
grad_step = 000268, loss = 0.001691
grad_step = 000269, loss = 0.001740
grad_step = 000270, loss = 0.001778
grad_step = 000271, loss = 0.001851
grad_step = 000272, loss = 0.001780
grad_step = 000273, loss = 0.001718
grad_step = 000274, loss = 0.001686
grad_step = 000275, loss = 0.001686
grad_step = 000276, loss = 0.001698
grad_step = 000277, loss = 0.001700
grad_step = 000278, loss = 0.001701
grad_step = 000279, loss = 0.001672
grad_step = 000280, loss = 0.001642
grad_step = 000281, loss = 0.001641
grad_step = 000282, loss = 0.001674
grad_step = 000283, loss = 0.001705
grad_step = 000284, loss = 0.001691
grad_step = 000285, loss = 0.001678
grad_step = 000286, loss = 0.001674
grad_step = 000287, loss = 0.001682
grad_step = 000288, loss = 0.001662
grad_step = 000289, loss = 0.001629
grad_step = 000290, loss = 0.001614
grad_step = 000291, loss = 0.001629
grad_step = 000292, loss = 0.001647
grad_step = 000293, loss = 0.001631
grad_step = 000294, loss = 0.001609
grad_step = 000295, loss = 0.001606
grad_step = 000296, loss = 0.001617
grad_step = 000297, loss = 0.001619
grad_step = 000298, loss = 0.001609
grad_step = 000299, loss = 0.001597
grad_step = 000300, loss = 0.001597
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001604
grad_step = 000302, loss = 0.001603
grad_step = 000303, loss = 0.001595
grad_step = 000304, loss = 0.001587
grad_step = 000305, loss = 0.001585
grad_step = 000306, loss = 0.001588
grad_step = 000307, loss = 0.001589
grad_step = 000308, loss = 0.001588
grad_step = 000309, loss = 0.001583
grad_step = 000310, loss = 0.001579
grad_step = 000311, loss = 0.001581
grad_step = 000312, loss = 0.001590
grad_step = 000313, loss = 0.001605
grad_step = 000314, loss = 0.001640
grad_step = 000315, loss = 0.001674
grad_step = 000316, loss = 0.001758
grad_step = 000317, loss = 0.001768
grad_step = 000318, loss = 0.001809
grad_step = 000319, loss = 0.001703
grad_step = 000320, loss = 0.001615
grad_step = 000321, loss = 0.001583
grad_step = 000322, loss = 0.001622
grad_step = 000323, loss = 0.001680
grad_step = 000324, loss = 0.001654
grad_step = 000325, loss = 0.001603
grad_step = 000326, loss = 0.001558
grad_step = 000327, loss = 0.001576
grad_step = 000328, loss = 0.001622
grad_step = 000329, loss = 0.001614
grad_step = 000330, loss = 0.001582
grad_step = 000331, loss = 0.001553
grad_step = 000332, loss = 0.001558
grad_step = 000333, loss = 0.001572
grad_step = 000334, loss = 0.001566
grad_step = 000335, loss = 0.001560
grad_step = 000336, loss = 0.001564
grad_step = 000337, loss = 0.001575
grad_step = 000338, loss = 0.001566
grad_step = 000339, loss = 0.001549
grad_step = 000340, loss = 0.001535
grad_step = 000341, loss = 0.001534
grad_step = 000342, loss = 0.001538
grad_step = 000343, loss = 0.001538
grad_step = 000344, loss = 0.001533
grad_step = 000345, loss = 0.001531
grad_step = 000346, loss = 0.001535
grad_step = 000347, loss = 0.001539
grad_step = 000348, loss = 0.001540
grad_step = 000349, loss = 0.001536
grad_step = 000350, loss = 0.001533
grad_step = 000351, loss = 0.001532
grad_step = 000352, loss = 0.001534
grad_step = 000353, loss = 0.001530
grad_step = 000354, loss = 0.001527
grad_step = 000355, loss = 0.001521
grad_step = 000356, loss = 0.001520
grad_step = 000357, loss = 0.001519
grad_step = 000358, loss = 0.001520
grad_step = 000359, loss = 0.001517
grad_step = 000360, loss = 0.001515
grad_step = 000361, loss = 0.001514
grad_step = 000362, loss = 0.001516
grad_step = 000363, loss = 0.001518
grad_step = 000364, loss = 0.001525
grad_step = 000365, loss = 0.001530
grad_step = 000366, loss = 0.001544
grad_step = 000367, loss = 0.001554
grad_step = 000368, loss = 0.001579
grad_step = 000369, loss = 0.001584
grad_step = 000370, loss = 0.001596
grad_step = 000371, loss = 0.001568
grad_step = 000372, loss = 0.001539
grad_step = 000373, loss = 0.001502
grad_step = 000374, loss = 0.001485
grad_step = 000375, loss = 0.001488
grad_step = 000376, loss = 0.001505
grad_step = 000377, loss = 0.001527
grad_step = 000378, loss = 0.001540
grad_step = 000379, loss = 0.001555
grad_step = 000380, loss = 0.001542
grad_step = 000381, loss = 0.001527
grad_step = 000382, loss = 0.001497
grad_step = 000383, loss = 0.001476
grad_step = 000384, loss = 0.001469
grad_step = 000385, loss = 0.001474
grad_step = 000386, loss = 0.001487
grad_step = 000387, loss = 0.001498
grad_step = 000388, loss = 0.001514
grad_step = 000389, loss = 0.001521
grad_step = 000390, loss = 0.001536
grad_step = 000391, loss = 0.001531
grad_step = 000392, loss = 0.001520
grad_step = 000393, loss = 0.001490
grad_step = 000394, loss = 0.001465
grad_step = 000395, loss = 0.001453
grad_step = 000396, loss = 0.001458
grad_step = 000397, loss = 0.001471
grad_step = 000398, loss = 0.001483
grad_step = 000399, loss = 0.001496
grad_step = 000400, loss = 0.001497
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001500
grad_step = 000402, loss = 0.001489
grad_step = 000403, loss = 0.001477
grad_step = 000404, loss = 0.001460
grad_step = 000405, loss = 0.001448
grad_step = 000406, loss = 0.001441
grad_step = 000407, loss = 0.001441
grad_step = 000408, loss = 0.001448
grad_step = 000409, loss = 0.001457
grad_step = 000410, loss = 0.001470
grad_step = 000411, loss = 0.001479
grad_step = 000412, loss = 0.001497
grad_step = 000413, loss = 0.001502
grad_step = 000414, loss = 0.001511
grad_step = 000415, loss = 0.001498
grad_step = 000416, loss = 0.001485
grad_step = 000417, loss = 0.001457
grad_step = 000418, loss = 0.001435
grad_step = 000419, loss = 0.001421
grad_step = 000420, loss = 0.001421
grad_step = 000421, loss = 0.001431
grad_step = 000422, loss = 0.001444
grad_step = 000423, loss = 0.001463
grad_step = 000424, loss = 0.001469
grad_step = 000425, loss = 0.001476
grad_step = 000426, loss = 0.001462
grad_step = 000427, loss = 0.001448
grad_step = 000428, loss = 0.001429
grad_step = 000429, loss = 0.001420
grad_step = 000430, loss = 0.001416
grad_step = 000431, loss = 0.001414
grad_step = 000432, loss = 0.001413
grad_step = 000433, loss = 0.001414
grad_step = 000434, loss = 0.001423
grad_step = 000435, loss = 0.001433
grad_step = 000436, loss = 0.001455
grad_step = 000437, loss = 0.001469
grad_step = 000438, loss = 0.001486
grad_step = 000439, loss = 0.001471
grad_step = 000440, loss = 0.001452
grad_step = 000441, loss = 0.001420
grad_step = 000442, loss = 0.001400
grad_step = 000443, loss = 0.001393
grad_step = 000444, loss = 0.001397
grad_step = 000445, loss = 0.001404
grad_step = 000446, loss = 0.001408
grad_step = 000447, loss = 0.001412
grad_step = 000448, loss = 0.001413
grad_step = 000449, loss = 0.001420
grad_step = 000450, loss = 0.001421
grad_step = 000451, loss = 0.001421
grad_step = 000452, loss = 0.001405
grad_step = 000453, loss = 0.001389
grad_step = 000454, loss = 0.001373
grad_step = 000455, loss = 0.001367
grad_step = 000456, loss = 0.001368
grad_step = 000457, loss = 0.001374
grad_step = 000458, loss = 0.001386
grad_step = 000459, loss = 0.001397
grad_step = 000460, loss = 0.001417
grad_step = 000461, loss = 0.001428
grad_step = 000462, loss = 0.001449
grad_step = 000463, loss = 0.001452
grad_step = 000464, loss = 0.001470
grad_step = 000465, loss = 0.001461
grad_step = 000466, loss = 0.001456
grad_step = 000467, loss = 0.001415
grad_step = 000468, loss = 0.001379
grad_step = 000469, loss = 0.001361
grad_step = 000470, loss = 0.001364
grad_step = 000471, loss = 0.001381
grad_step = 000472, loss = 0.001393
grad_step = 000473, loss = 0.001403
grad_step = 000474, loss = 0.001383
grad_step = 000475, loss = 0.001357
grad_step = 000476, loss = 0.001337
grad_step = 000477, loss = 0.001341
grad_step = 000478, loss = 0.001355
grad_step = 000479, loss = 0.001356
grad_step = 000480, loss = 0.001347
grad_step = 000481, loss = 0.001340
grad_step = 000482, loss = 0.001343
grad_step = 000483, loss = 0.001344
grad_step = 000484, loss = 0.001345
grad_step = 000485, loss = 0.001349
grad_step = 000486, loss = 0.001365
grad_step = 000487, loss = 0.001377
grad_step = 000488, loss = 0.001391
grad_step = 000489, loss = 0.001390
grad_step = 000490, loss = 0.001398
grad_step = 000491, loss = 0.001388
grad_step = 000492, loss = 0.001387
grad_step = 000493, loss = 0.001367
grad_step = 000494, loss = 0.001358
grad_step = 000495, loss = 0.001346
grad_step = 000496, loss = 0.001343
grad_step = 000497, loss = 0.001335
grad_step = 000498, loss = 0.001325
grad_step = 000499, loss = 0.001316
grad_step = 000500, loss = 0.001310
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001310
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

  date_run                              2020-05-25 23:50:21.820379
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.215634
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-25 23:50:21.826121
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.106755
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-25 23:50:21.832685
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.137337
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-25 23:50:21.838868
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.622173
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fae1267c860> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 23:50:38.315994
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-25 23:50:38.319588
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-25 23:50:38.323475
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-25 23:50:38.327122
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
0   2020-05-25 23:50:00.767821  ...    mean_absolute_error
1   2020-05-25 23:50:00.772687  ...     mean_squared_error
2   2020-05-25 23:50:00.776421  ...  median_absolute_error
3   2020-05-25 23:50:00.779872  ...               r2_score
4   2020-05-25 23:50:21.820379  ...    mean_absolute_error
5   2020-05-25 23:50:21.826121  ...     mean_squared_error
6   2020-05-25 23:50:21.832685  ...  median_absolute_error
7   2020-05-25 23:50:21.838868  ...               r2_score
8   2020-05-25 23:50:38.315994  ...    mean_absolute_error
9   2020-05-25 23:50:38.319588  ...     mean_squared_error
10  2020-05-25 23:50:38.323475  ...  median_absolute_error
11  2020-05-25 23:50:38.327122  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 118, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

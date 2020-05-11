
  test_cli /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_cli', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_cli 

  # Testing Command Line System   





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/314d31f204bf54d585069061e7912b076f4501c5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '314d31f204bf54d585069061e7912b076f4501c5', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/314d31f204bf54d585069061e7912b076f4501c5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/314d31f204bf54d585069061e7912b076f4501c5

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
model_tf.1_lstm
model_sklearn.model_lightgbm
model_sklearn.model_sklearn
model_tch.pplm
model_tch.transformer_sentence
model_tch.nbeats
model_tch.textcnn
model_tch.mlp
model_tchtorch_vae
model_tch.03_nbeats_dataloader
model_tch.torchhub
model_tch.transformer_classifier
model_tch.matchzoo_models
model_gluon.gluonts_model
model_gluon.fb_prophet
model_gluon.gluon_automl
model_keras.charcnn
model_keras.charcnn_zhang
model_keras.namentity_crm_bilstm_dataloader
model_keras.01_deepctr
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
fit

  ##### Load JSON /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.json 

  ##### Init model_tf.1_lstm {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'} 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  ##### Fit <mlmodels.model_tf.1_lstm.Model object at 0x7f079592fe48> 
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

  ##### Save <tensorflow.python.client.session.Session object at 0x7f078fbaab70> 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm/', 'model_uri': 'model_tf.1_lstm'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest_1lstm//model//model.ckpt





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
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 531, in main
    predict_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 442, in predict_cli
    model, session = load(module, load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 156, in load
    return module.load(load_pars, **kwarg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 445, in load_tf
    saver      = tf.compat.v1.train.Saver()  
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f4ab6d8c208> 

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
 [ 0.0110343  -0.02451492  0.04981149  0.01866864  0.04903439 -0.02702744]
 [ 0.16347481 -0.14844984  0.00694107 -0.03160354  0.22309062 -0.06424998]
 [-0.09675784  0.23498903 -0.05461435 -0.43187249  0.074541    0.28243932]
 [-0.16257398 -0.30152625  0.16832681 -0.09679507  0.10186034  0.03456865]
 [-0.12335577  0.34516808 -0.02022463  0.21600145  0.46115512  0.01838106]
 [ 0.13282973 -0.21083553  0.23651971  0.03480964  0.15611365 -0.1656394 ]
 [ 0.19779913  0.13741305 -0.12738284 -0.19864632  0.07571667 -0.02473857]
 [ 0.23964192  0.15599294  0.10831276  0.22137249  0.01522447 -0.18729937]
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
{'loss': 0.5200747922062874, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
{'loss': 0.5285084918141365, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





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

  <mlmodels.example.custom_model.1_lstm.Model object at 0x7efe773e6320> 

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
 [ 0.03519234 -0.02620151 -0.0973173   0.01444409  0.13105899 -0.07028428]
 [-0.00580436 -0.04289605  0.01294695 -0.0455016  -0.20999481  0.10726314]
 [ 0.07472071  0.16335018  0.08570805 -0.04976185 -0.02022426 -0.00426972]
 [-0.11050485  0.27826613  0.28223991  0.08146109 -0.11092168  0.15799661]
 [ 0.23899047  0.21377932  0.30345702  0.22171223  0.1339968  -0.35410416]
 [ 0.00872441  0.45233306  0.1701678  -0.32110381 -0.17566635  0.57488579]
 [-0.37475032 -0.38417947 -0.00877786 -0.24890102 -0.0641991   0.56895602]
 [-0.08960217  0.30542308  0.31882274 -0.01980869  0.04974316 -0.45012522]
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
{'loss': 0.6038771867752075, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
{'loss': 0.48637905716896057, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





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
[32m[I 2020-05-11 06:40:07,098][0m Finished trial#0 resulted in value: 0.8134631216526031. Current best value is 0.8134631216526031 with parameters: {'learning_rate': 0.02459463022485579, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-11 06:40:09,303][0m Finished trial#1 resulted in value: 0.314064085483551. Current best value is 0.314064085483551 with parameters: {'learning_rate': 0.0019496627249157078, 'num_layers': 4, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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

  #############  OPTIMIZATION End ############### 

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.0019496627249157078, 'num_layers': 4, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 5, 'epoch': 2, 'best_value': 0.314064085483551, 'model_name': None} 





 ************************************************************************************************************************
ml_optim --do test   --model_uri model_tf.1_lstm   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-11 06:40:18,004][0m Finished trial#0 resulted in value: 0.3155150040984154. Current best value is 0.3155150040984154 with parameters: {'learning_rate': 0.0010724491372579561, 'num_layers': 4, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-11 06:40:19,918][0m Finished trial#1 resulted in value: 0.29138489812612534. Current best value is 0.29138489812612534 with parameters: {'learning_rate': 0.0016951778671498337, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff3ef38e128> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:40:39.756074
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 06:40:39.760585
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 06:40:39.765100
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 06:40:39.768710
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff3de92d198> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355357.3125
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 284241.1250
Epoch 3/10

1/1 [==============================] - 0s 105ms/step - loss: 179053.2344
Epoch 4/10

1/1 [==============================] - 0s 116ms/step - loss: 96424.9844
Epoch 5/10

1/1 [==============================] - 0s 119ms/step - loss: 51972.6836
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 30536.5527
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 19620.7773
Epoch 8/10

1/1 [==============================] - 0s 99ms/step - loss: 13565.2666
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 9948.3232
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 7673.2129

  #### Inference Need return ypred, ytrue ######################### 
[[ 9.61314976e-01 -1.09144497e+00  1.94459766e-01  5.91898084e-01
  -1.44732344e+00 -3.94281209e-01 -5.21972179e-01 -9.23075438e-01
  -1.96543902e-01 -3.80228877e-01 -5.75128615e-01  7.81768858e-01
  -8.32163751e-01 -1.25243858e-01 -1.07533836e+00 -4.73123103e-01
  -1.82485819e-01 -1.98363543e-01 -2.13883847e-01  6.75727963e-01
  -6.63886189e-01 -1.01758480e+00 -3.44286382e-01  1.12392604e-02
   3.31480891e-01 -5.73912263e-02 -1.57442391e-01 -1.53117132e+00
  -1.83466002e-01  4.94524539e-02  5.23466229e-01 -1.08069491e+00
   5.38573146e-01 -1.47893578e-01  6.37098789e-01 -3.94434512e-01
   1.19729042e-02 -6.73594594e-01  8.27521682e-02 -1.27804863e+00
  -1.09761739e+00  8.72767329e-01  1.03501010e+00  3.52694467e-02
  -3.13055485e-01 -9.52317238e-01  2.47353286e-01  6.05177879e-03
   1.10550070e+00  5.23396730e-01 -8.82455349e-01  1.17104217e-01
   5.36397994e-02  4.23349917e-01  1.36396980e+00  2.38506600e-01
   6.49011791e-01 -3.18803459e-01 -8.55678797e-01 -1.54913390e+00
  -6.26808405e-02  8.16640258e-01  8.02187741e-01 -2.77371019e-01
   8.89187813e-01 -1.50428021e+00 -2.10008249e-01 -4.63094711e-01
  -8.47551644e-01 -2.02206731e-01  6.10690236e-01  9.47916657e-02
  -6.86862111e-01  9.61140275e-01  6.84280276e-01 -6.95523739e-01
   3.59336883e-01  8.96808267e-01 -4.44663793e-01 -1.55629504e+00
  -9.29150820e-01  6.04091048e-01  2.27431297e-01 -5.74315429e-01
   1.88862157e+00 -9.96847749e-01  1.01902425e+00  1.03638732e+00
   3.70729923e-01 -9.75244939e-01 -1.06522381e+00  4.47558284e-01
  -4.42891449e-01  8.03275943e-01 -6.60416543e-01  1.19454622e+00
  -8.33393335e-01 -5.02336562e-01 -2.27782726e-01 -9.82899785e-01
   6.96461558e-01  1.65635931e+00  1.48623085e+00 -1.18314326e+00
  -8.24535489e-02  2.38149524e-01  1.99980223e+00  1.28955698e+00
  -2.40667403e-01  6.00440025e-01 -1.22928238e+00  1.04146707e+00
  -7.78293312e-02 -1.05497181e+00 -2.76707679e-01  1.66036975e+00
  -1.08866036e+00 -2.82579660e-01 -7.69742906e-01 -6.57592952e-01
   8.46766457e-02  7.78038359e+00  7.70833302e+00  7.85902119e+00
   7.52405167e+00  7.42237329e+00  6.79531717e+00  5.88207531e+00
   7.33892155e+00  5.82332659e+00  7.06939268e+00  8.39153576e+00
   7.16016912e+00  6.57114410e+00  7.34871435e+00  7.29555130e+00
   8.30634022e+00  6.83054733e+00  8.34548473e+00  9.05870724e+00
   6.90789413e+00  6.53985357e+00  7.21782589e+00  8.66399956e+00
   7.75160313e+00  6.86121702e+00  7.11775255e+00  7.00856161e+00
   7.69117737e+00  6.26943684e+00  6.20658350e+00  5.51967192e+00
   7.52010870e+00  6.81258440e+00  7.40227318e+00  8.83016014e+00
   6.73042536e+00  7.58937263e+00  8.49049282e+00  6.35856724e+00
   6.56370640e+00  6.58364916e+00  8.80213451e+00  8.22568226e+00
   6.37820625e+00  6.83307838e+00  6.60058212e+00  6.18186712e+00
   6.82642508e+00  8.62930489e+00  8.38640499e+00  8.10018921e+00
   8.23420429e+00  7.18928146e+00  8.22139072e+00  8.44399071e+00
   9.07323074e+00  6.90539455e+00  6.86144543e+00  6.49833870e+00
   2.55260110e-01  9.89008784e-01  1.98723686e+00  4.76980329e-01
   1.10014951e+00  5.86808681e-01  3.10369849e-01  1.20774388e+00
   1.18338466e-01  1.59349501e+00  1.97084188e-01  1.01671267e+00
   6.01021588e-01  4.13328886e-01  7.04731822e-01  1.20679724e+00
   8.67460549e-01  9.42272007e-01  2.70101309e-01  3.48022223e-01
   6.25971496e-01  2.22716713e+00  2.69128609e+00  3.44040453e-01
   6.61307216e-01  8.91573668e-01  6.06648505e-01  2.31659174e+00
   8.10138822e-01  1.15232289e+00  2.71942663e+00  1.14674914e+00
   3.47813189e-01  4.70999420e-01  4.29269016e-01  1.10263908e+00
   3.35942686e-01  6.32857263e-01  1.16250968e+00  1.27048135e+00
   1.87848568e-01  4.47184563e-01  1.19230402e+00  1.01327217e+00
   1.39954174e+00  1.33486080e+00  1.21196389e+00  1.99649143e+00
   1.98095417e+00  8.97283316e-01  4.51746106e-01  8.13118219e-01
   9.57057118e-01  5.34863770e-01  3.78378749e-01  1.27399755e+00
   7.94396222e-01  4.30134892e-01  5.32360554e-01  1.46501899e+00
   8.09458613e-01  1.32528520e+00  2.51965332e+00  1.43829584e+00
   8.47425342e-01  5.06154299e-01  1.83347344e-01  1.65312886e+00
   2.36134005e+00  2.63786972e-01  2.85302222e-01  6.02132440e-01
   2.36069059e+00  1.47662771e+00  8.15116942e-01  7.93183982e-01
   1.49230480e+00  1.28442025e+00  1.23715079e+00  4.42476630e-01
   7.29973495e-01  2.13905096e+00  1.41717708e+00  8.02574992e-01
   7.17214108e-01  2.61447406e+00  1.08995187e+00  6.79437637e-01
   6.26352429e-01  2.82400250e-01  1.02238238e+00  7.39296317e-01
   9.41090882e-01  6.22948050e-01  9.10935521e-01  1.97738290e+00
   4.43586707e-01  4.71852064e-01  1.62771797e+00  8.94459009e-01
   1.85107946e+00  1.10964286e+00  2.05817604e+00  5.01208842e-01
   4.22950923e-01  4.40049350e-01  5.89623630e-01  6.08985484e-01
   3.80665958e-01  1.18255913e-01  2.52442265e+00  1.04003513e+00
   1.10289013e+00  3.14669371e-01  5.20935476e-01  1.18659163e+00
   2.43583179e+00  1.72929239e+00  8.72422993e-01  1.81036425e+00
   9.43105817e-02  7.43823719e+00  7.87653732e+00  8.27054691e+00
   7.99122143e+00  8.64516830e+00  8.36182594e+00  7.89465952e+00
   8.21992493e+00  7.15240431e+00  6.89248610e+00  5.87508249e+00
   7.07595015e+00  6.60084438e+00  7.58603716e+00  6.85023022e+00
   8.76723766e+00  7.98824978e+00  7.49202061e+00  9.04009628e+00
   8.24632931e+00  7.41453743e+00  8.29358101e+00  7.81172180e+00
   7.48342085e+00  7.17651033e+00  8.51142693e+00  7.75059128e+00
   8.26374531e+00  7.56047153e+00  7.66925764e+00  8.35009766e+00
   7.52584076e+00  7.75988054e+00  8.27476120e+00  7.59644318e+00
   7.83683300e+00  7.85395050e+00  8.15372849e+00  6.63174820e+00
   7.22503710e+00  6.84716558e+00  8.19693279e+00  7.62954473e+00
   8.46130943e+00  6.43844080e+00  7.45871019e+00  8.14605713e+00
   8.80580521e+00  7.74774408e+00  7.51956892e+00  7.33914709e+00
   8.39684582e+00  7.57143784e+00  7.34082127e+00  7.71144056e+00
   7.41112566e+00  6.93751001e+00  7.98199749e+00  8.31715012e+00
  -1.94765520e+00 -9.73427677e+00  4.48339176e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:40:49.285253
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.6394
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 06:40:49.289729
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                      8976
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 06:40:49.293909
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.4811
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 06:40:49.297715
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -802.864
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140685050779464
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140684099738256
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140684099738760
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140684099739264
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140684099760312
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140684099760816

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff3ef56efd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.647618
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.618051
grad_step = 000002, loss = 0.599056
grad_step = 000003, loss = 0.578007
grad_step = 000004, loss = 0.555062
grad_step = 000005, loss = 0.532030
grad_step = 000006, loss = 0.514116
grad_step = 000007, loss = 0.503266
grad_step = 000008, loss = 0.486486
grad_step = 000009, loss = 0.464547
grad_step = 000010, loss = 0.443660
grad_step = 000011, loss = 0.425692
grad_step = 000012, loss = 0.409977
grad_step = 000013, loss = 0.394154
grad_step = 000014, loss = 0.378442
grad_step = 000015, loss = 0.362643
grad_step = 000016, loss = 0.347247
grad_step = 000017, loss = 0.333411
grad_step = 000018, loss = 0.320831
grad_step = 000019, loss = 0.309130
grad_step = 000020, loss = 0.299933
grad_step = 000021, loss = 0.289093
grad_step = 000022, loss = 0.276519
grad_step = 000023, loss = 0.265008
grad_step = 000024, loss = 0.254745
grad_step = 000025, loss = 0.244908
grad_step = 000026, loss = 0.235025
grad_step = 000027, loss = 0.224998
grad_step = 000028, loss = 0.215020
grad_step = 000029, loss = 0.205327
grad_step = 000030, loss = 0.196108
grad_step = 000031, loss = 0.187562
grad_step = 000032, loss = 0.179202
grad_step = 000033, loss = 0.170779
grad_step = 000034, loss = 0.162648
grad_step = 000035, loss = 0.154834
grad_step = 000036, loss = 0.147389
grad_step = 000037, loss = 0.140286
grad_step = 000038, loss = 0.133419
grad_step = 000039, loss = 0.126909
grad_step = 000040, loss = 0.120707
grad_step = 000041, loss = 0.114861
grad_step = 000042, loss = 0.109271
grad_step = 000043, loss = 0.103764
grad_step = 000044, loss = 0.098478
grad_step = 000045, loss = 0.093515
grad_step = 000046, loss = 0.088919
grad_step = 000047, loss = 0.084573
grad_step = 000048, loss = 0.080429
grad_step = 000049, loss = 0.076427
grad_step = 000050, loss = 0.072613
grad_step = 000051, loss = 0.069057
grad_step = 000052, loss = 0.065683
grad_step = 000053, loss = 0.062482
grad_step = 000054, loss = 0.059433
grad_step = 000055, loss = 0.056574
grad_step = 000056, loss = 0.053861
grad_step = 000057, loss = 0.051291
grad_step = 000058, loss = 0.048833
grad_step = 000059, loss = 0.046509
grad_step = 000060, loss = 0.044321
grad_step = 000061, loss = 0.042263
grad_step = 000062, loss = 0.040293
grad_step = 000063, loss = 0.038429
grad_step = 000064, loss = 0.036656
grad_step = 000065, loss = 0.034987
grad_step = 000066, loss = 0.033397
grad_step = 000067, loss = 0.031881
grad_step = 000068, loss = 0.030435
grad_step = 000069, loss = 0.029074
grad_step = 000070, loss = 0.027789
grad_step = 000071, loss = 0.026565
grad_step = 000072, loss = 0.025397
grad_step = 000073, loss = 0.024280
grad_step = 000074, loss = 0.023221
grad_step = 000075, loss = 0.022210
grad_step = 000076, loss = 0.021249
grad_step = 000077, loss = 0.020334
grad_step = 000078, loss = 0.019463
grad_step = 000079, loss = 0.018634
grad_step = 000080, loss = 0.017840
grad_step = 000081, loss = 0.017082
grad_step = 000082, loss = 0.016362
grad_step = 000083, loss = 0.015677
grad_step = 000084, loss = 0.015023
grad_step = 000085, loss = 0.014396
grad_step = 000086, loss = 0.013799
grad_step = 000087, loss = 0.013231
grad_step = 000088, loss = 0.012691
grad_step = 000089, loss = 0.012177
grad_step = 000090, loss = 0.011686
grad_step = 000091, loss = 0.011219
grad_step = 000092, loss = 0.010774
grad_step = 000093, loss = 0.010350
grad_step = 000094, loss = 0.009946
grad_step = 000095, loss = 0.009561
grad_step = 000096, loss = 0.009196
grad_step = 000097, loss = 0.008847
grad_step = 000098, loss = 0.008515
grad_step = 000099, loss = 0.008198
grad_step = 000100, loss = 0.007896
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.007608
grad_step = 000102, loss = 0.007333
grad_step = 000103, loss = 0.007071
grad_step = 000104, loss = 0.006821
grad_step = 000105, loss = 0.006583
grad_step = 000106, loss = 0.006356
grad_step = 000107, loss = 0.006138
grad_step = 000108, loss = 0.005931
grad_step = 000109, loss = 0.005733
grad_step = 000110, loss = 0.005544
grad_step = 000111, loss = 0.005363
grad_step = 000112, loss = 0.005191
grad_step = 000113, loss = 0.005026
grad_step = 000114, loss = 0.004869
grad_step = 000115, loss = 0.004720
grad_step = 000116, loss = 0.004577
grad_step = 000117, loss = 0.004440
grad_step = 000118, loss = 0.004311
grad_step = 000119, loss = 0.004187
grad_step = 000120, loss = 0.004068
grad_step = 000121, loss = 0.003955
grad_step = 000122, loss = 0.003847
grad_step = 000123, loss = 0.003745
grad_step = 000124, loss = 0.003648
grad_step = 000125, loss = 0.003555
grad_step = 000126, loss = 0.003466
grad_step = 000127, loss = 0.003381
grad_step = 000128, loss = 0.003301
grad_step = 000129, loss = 0.003224
grad_step = 000130, loss = 0.003151
grad_step = 000131, loss = 0.003082
grad_step = 000132, loss = 0.003017
grad_step = 000133, loss = 0.002956
grad_step = 000134, loss = 0.002899
grad_step = 000135, loss = 0.002849
grad_step = 000136, loss = 0.002806
grad_step = 000137, loss = 0.002761
grad_step = 000138, loss = 0.002718
grad_step = 000139, loss = 0.002675
grad_step = 000140, loss = 0.002623
grad_step = 000141, loss = 0.002571
grad_step = 000142, loss = 0.002528
grad_step = 000143, loss = 0.002489
grad_step = 000144, loss = 0.002453
grad_step = 000145, loss = 0.002426
grad_step = 000146, loss = 0.002402
grad_step = 000147, loss = 0.002378
grad_step = 000148, loss = 0.002353
grad_step = 000149, loss = 0.002332
grad_step = 000150, loss = 0.002307
grad_step = 000151, loss = 0.002280
grad_step = 000152, loss = 0.002254
grad_step = 000153, loss = 0.002230
grad_step = 000154, loss = 0.002204
grad_step = 000155, loss = 0.002182
grad_step = 000156, loss = 0.002165
grad_step = 000157, loss = 0.002149
grad_step = 000158, loss = 0.002134
grad_step = 000159, loss = 0.002123
grad_step = 000160, loss = 0.002114
grad_step = 000161, loss = 0.002106
grad_step = 000162, loss = 0.002105
grad_step = 000163, loss = 0.002111
grad_step = 000164, loss = 0.002130
grad_step = 000165, loss = 0.002142
grad_step = 000166, loss = 0.002152
grad_step = 000167, loss = 0.002127
grad_step = 000168, loss = 0.002072
grad_step = 000169, loss = 0.002022
grad_step = 000170, loss = 0.002013
grad_step = 000171, loss = 0.002033
grad_step = 000172, loss = 0.002049
grad_step = 000173, loss = 0.002041
grad_step = 000174, loss = 0.002010
grad_step = 000175, loss = 0.001981
grad_step = 000176, loss = 0.001973
grad_step = 000177, loss = 0.001984
grad_step = 000178, loss = 0.001996
grad_step = 000179, loss = 0.001988
grad_step = 000180, loss = 0.001969
grad_step = 000181, loss = 0.001952
grad_step = 000182, loss = 0.001946
grad_step = 000183, loss = 0.001950
grad_step = 000184, loss = 0.001955
grad_step = 000185, loss = 0.001952
grad_step = 000186, loss = 0.001940
grad_step = 000187, loss = 0.001929
grad_step = 000188, loss = 0.001925
grad_step = 000189, loss = 0.001926
grad_step = 000190, loss = 0.001927
grad_step = 000191, loss = 0.001927
grad_step = 000192, loss = 0.001923
grad_step = 000193, loss = 0.001916
grad_step = 000194, loss = 0.001909
grad_step = 000195, loss = 0.001905
grad_step = 000196, loss = 0.001903
grad_step = 000197, loss = 0.001903
grad_step = 000198, loss = 0.001903
grad_step = 000199, loss = 0.001903
grad_step = 000200, loss = 0.001901
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001898
grad_step = 000202, loss = 0.001894
grad_step = 000203, loss = 0.001890
grad_step = 000204, loss = 0.001887
grad_step = 000205, loss = 0.001883
grad_step = 000206, loss = 0.001880
grad_step = 000207, loss = 0.001878
grad_step = 000208, loss = 0.001875
grad_step = 000209, loss = 0.001873
grad_step = 000210, loss = 0.001870
grad_step = 000211, loss = 0.001868
grad_step = 000212, loss = 0.001866
grad_step = 000213, loss = 0.001863
grad_step = 000214, loss = 0.001861
grad_step = 000215, loss = 0.001861
grad_step = 000216, loss = 0.001861
grad_step = 000217, loss = 0.001872
grad_step = 000218, loss = 0.001903
grad_step = 000219, loss = 0.001987
grad_step = 000220, loss = 0.002051
grad_step = 000221, loss = 0.002131
grad_step = 000222, loss = 0.002046
grad_step = 000223, loss = 0.001998
grad_step = 000224, loss = 0.001985
grad_step = 000225, loss = 0.001878
grad_step = 000226, loss = 0.001875
grad_step = 000227, loss = 0.001969
grad_step = 000228, loss = 0.001923
grad_step = 000229, loss = 0.001829
grad_step = 000230, loss = 0.001852
grad_step = 000231, loss = 0.001902
grad_step = 000232, loss = 0.001864
grad_step = 000233, loss = 0.001812
grad_step = 000234, loss = 0.001838
grad_step = 000235, loss = 0.001863
grad_step = 000236, loss = 0.001827
grad_step = 000237, loss = 0.001802
grad_step = 000238, loss = 0.001819
grad_step = 000239, loss = 0.001823
grad_step = 000240, loss = 0.001792
grad_step = 000241, loss = 0.001780
grad_step = 000242, loss = 0.001798
grad_step = 000243, loss = 0.001794
grad_step = 000244, loss = 0.001768
grad_step = 000245, loss = 0.001767
grad_step = 000246, loss = 0.001777
grad_step = 000247, loss = 0.001765
grad_step = 000248, loss = 0.001751
grad_step = 000249, loss = 0.001755
grad_step = 000250, loss = 0.001762
grad_step = 000251, loss = 0.001762
grad_step = 000252, loss = 0.001775
grad_step = 000253, loss = 0.001819
grad_step = 000254, loss = 0.001898
grad_step = 000255, loss = 0.001934
grad_step = 000256, loss = 0.001940
grad_step = 000257, loss = 0.001815
grad_step = 000258, loss = 0.001731
grad_step = 000259, loss = 0.001747
grad_step = 000260, loss = 0.001795
grad_step = 000261, loss = 0.001795
grad_step = 000262, loss = 0.001737
grad_step = 000263, loss = 0.001715
grad_step = 000264, loss = 0.001744
grad_step = 000265, loss = 0.001753
grad_step = 000266, loss = 0.001723
grad_step = 000267, loss = 0.001697
grad_step = 000268, loss = 0.001708
grad_step = 000269, loss = 0.001728
grad_step = 000270, loss = 0.001711
grad_step = 000271, loss = 0.001682
grad_step = 000272, loss = 0.001684
grad_step = 000273, loss = 0.001703
grad_step = 000274, loss = 0.001699
grad_step = 000275, loss = 0.001675
grad_step = 000276, loss = 0.001665
grad_step = 000277, loss = 0.001675
grad_step = 000278, loss = 0.001679
grad_step = 000279, loss = 0.001671
grad_step = 000280, loss = 0.001662
grad_step = 000281, loss = 0.001659
grad_step = 000282, loss = 0.001656
grad_step = 000283, loss = 0.001649
grad_step = 000284, loss = 0.001646
grad_step = 000285, loss = 0.001650
grad_step = 000286, loss = 0.001652
grad_step = 000287, loss = 0.001646
grad_step = 000288, loss = 0.001641
grad_step = 000289, loss = 0.001637
grad_step = 000290, loss = 0.001636
grad_step = 000291, loss = 0.001631
grad_step = 000292, loss = 0.001625
grad_step = 000293, loss = 0.001621
grad_step = 000294, loss = 0.001618
grad_step = 000295, loss = 0.001615
grad_step = 000296, loss = 0.001610
grad_step = 000297, loss = 0.001606
grad_step = 000298, loss = 0.001604
grad_step = 000299, loss = 0.001604
grad_step = 000300, loss = 0.001604
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001608
grad_step = 000302, loss = 0.001618
grad_step = 000303, loss = 0.001635
grad_step = 000304, loss = 0.001665
grad_step = 000305, loss = 0.001686
grad_step = 000306, loss = 0.001706
grad_step = 000307, loss = 0.001670
grad_step = 000308, loss = 0.001619
grad_step = 000309, loss = 0.001571
grad_step = 000310, loss = 0.001567
grad_step = 000311, loss = 0.001592
grad_step = 000312, loss = 0.001602
grad_step = 000313, loss = 0.001589
grad_step = 000314, loss = 0.001566
grad_step = 000315, loss = 0.001555
grad_step = 000316, loss = 0.001559
grad_step = 000317, loss = 0.001565
grad_step = 000318, loss = 0.001569
grad_step = 000319, loss = 0.001564
grad_step = 000320, loss = 0.001553
grad_step = 000321, loss = 0.001540
grad_step = 000322, loss = 0.001529
grad_step = 000323, loss = 0.001527
grad_step = 000324, loss = 0.001533
grad_step = 000325, loss = 0.001544
grad_step = 000326, loss = 0.001556
grad_step = 000327, loss = 0.001574
grad_step = 000328, loss = 0.001588
grad_step = 000329, loss = 0.001611
grad_step = 000330, loss = 0.001610
grad_step = 000331, loss = 0.001598
grad_step = 000332, loss = 0.001550
grad_step = 000333, loss = 0.001514
grad_step = 000334, loss = 0.001516
grad_step = 000335, loss = 0.001537
grad_step = 000336, loss = 0.001557
grad_step = 000337, loss = 0.001571
grad_step = 000338, loss = 0.001571
grad_step = 000339, loss = 0.001546
grad_step = 000340, loss = 0.001523
grad_step = 000341, loss = 0.001518
grad_step = 000342, loss = 0.001512
grad_step = 000343, loss = 0.001502
grad_step = 000344, loss = 0.001508
grad_step = 000345, loss = 0.001521
grad_step = 000346, loss = 0.001521
grad_step = 000347, loss = 0.001521
grad_step = 000348, loss = 0.001522
grad_step = 000349, loss = 0.001517
grad_step = 000350, loss = 0.001501
grad_step = 000351, loss = 0.001492
grad_step = 000352, loss = 0.001488
grad_step = 000353, loss = 0.001483
grad_step = 000354, loss = 0.001482
grad_step = 000355, loss = 0.001487
grad_step = 000356, loss = 0.001489
grad_step = 000357, loss = 0.001486
grad_step = 000358, loss = 0.001485
grad_step = 000359, loss = 0.001486
grad_step = 000360, loss = 0.001485
grad_step = 000361, loss = 0.001482
grad_step = 000362, loss = 0.001483
grad_step = 000363, loss = 0.001484
grad_step = 000364, loss = 0.001485
grad_step = 000365, loss = 0.001483
grad_step = 000366, loss = 0.001484
grad_step = 000367, loss = 0.001483
grad_step = 000368, loss = 0.001481
grad_step = 000369, loss = 0.001479
grad_step = 000370, loss = 0.001476
grad_step = 000371, loss = 0.001472
grad_step = 000372, loss = 0.001468
grad_step = 000373, loss = 0.001464
grad_step = 000374, loss = 0.001462
grad_step = 000375, loss = 0.001460
grad_step = 000376, loss = 0.001460
grad_step = 000377, loss = 0.001460
grad_step = 000378, loss = 0.001463
grad_step = 000379, loss = 0.001467
grad_step = 000380, loss = 0.001477
grad_step = 000381, loss = 0.001492
grad_step = 000382, loss = 0.001521
grad_step = 000383, loss = 0.001553
grad_step = 000384, loss = 0.001596
grad_step = 000385, loss = 0.001613
grad_step = 000386, loss = 0.001584
grad_step = 000387, loss = 0.001506
grad_step = 000388, loss = 0.001449
grad_step = 000389, loss = 0.001440
grad_step = 000390, loss = 0.001475
grad_step = 000391, loss = 0.001512
grad_step = 000392, loss = 0.001500
grad_step = 000393, loss = 0.001471
grad_step = 000394, loss = 0.001443
grad_step = 000395, loss = 0.001433
grad_step = 000396, loss = 0.001443
grad_step = 000397, loss = 0.001464
grad_step = 000398, loss = 0.001475
grad_step = 000399, loss = 0.001467
grad_step = 000400, loss = 0.001450
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001434
grad_step = 000402, loss = 0.001428
grad_step = 000403, loss = 0.001435
grad_step = 000404, loss = 0.001448
grad_step = 000405, loss = 0.001454
grad_step = 000406, loss = 0.001447
grad_step = 000407, loss = 0.001439
grad_step = 000408, loss = 0.001429
grad_step = 000409, loss = 0.001426
grad_step = 000410, loss = 0.001433
grad_step = 000411, loss = 0.001438
grad_step = 000412, loss = 0.001435
grad_step = 000413, loss = 0.001425
grad_step = 000414, loss = 0.001416
grad_step = 000415, loss = 0.001407
grad_step = 000416, loss = 0.001407
grad_step = 000417, loss = 0.001414
grad_step = 000418, loss = 0.001422
grad_step = 000419, loss = 0.001432
grad_step = 000420, loss = 0.001437
grad_step = 000421, loss = 0.001442
grad_step = 000422, loss = 0.001443
grad_step = 000423, loss = 0.001446
grad_step = 000424, loss = 0.001450
grad_step = 000425, loss = 0.001450
grad_step = 000426, loss = 0.001450
grad_step = 000427, loss = 0.001437
grad_step = 000428, loss = 0.001419
grad_step = 000429, loss = 0.001395
grad_step = 000430, loss = 0.001377
grad_step = 000431, loss = 0.001370
grad_step = 000432, loss = 0.001374
grad_step = 000433, loss = 0.001383
grad_step = 000434, loss = 0.001391
grad_step = 000435, loss = 0.001396
grad_step = 000436, loss = 0.001397
grad_step = 000437, loss = 0.001397
grad_step = 000438, loss = 0.001400
grad_step = 000439, loss = 0.001408
grad_step = 000440, loss = 0.001423
grad_step = 000441, loss = 0.001435
grad_step = 000442, loss = 0.001450
grad_step = 000443, loss = 0.001447
grad_step = 000444, loss = 0.001436
grad_step = 000445, loss = 0.001405
grad_step = 000446, loss = 0.001374
grad_step = 000447, loss = 0.001353
grad_step = 000448, loss = 0.001350
grad_step = 000449, loss = 0.001357
grad_step = 000450, loss = 0.001368
grad_step = 000451, loss = 0.001378
grad_step = 000452, loss = 0.001381
grad_step = 000453, loss = 0.001383
grad_step = 000454, loss = 0.001380
grad_step = 000455, loss = 0.001378
grad_step = 000456, loss = 0.001375
grad_step = 000457, loss = 0.001373
grad_step = 000458, loss = 0.001371
grad_step = 000459, loss = 0.001365
grad_step = 000460, loss = 0.001359
grad_step = 000461, loss = 0.001349
grad_step = 000462, loss = 0.001340
grad_step = 000463, loss = 0.001332
grad_step = 000464, loss = 0.001326
grad_step = 000465, loss = 0.001324
grad_step = 000466, loss = 0.001324
grad_step = 000467, loss = 0.001325
grad_step = 000468, loss = 0.001328
grad_step = 000469, loss = 0.001331
grad_step = 000470, loss = 0.001334
grad_step = 000471, loss = 0.001339
grad_step = 000472, loss = 0.001344
grad_step = 000473, loss = 0.001350
grad_step = 000474, loss = 0.001355
grad_step = 000475, loss = 0.001359
grad_step = 000476, loss = 0.001360
grad_step = 000477, loss = 0.001359
grad_step = 000478, loss = 0.001355
grad_step = 000479, loss = 0.001347
grad_step = 000480, loss = 0.001342
grad_step = 000481, loss = 0.001337
grad_step = 000482, loss = 0.001340
grad_step = 000483, loss = 0.001350
grad_step = 000484, loss = 0.001364
grad_step = 000485, loss = 0.001377
grad_step = 000486, loss = 0.001376
grad_step = 000487, loss = 0.001362
grad_step = 000488, loss = 0.001341
grad_step = 000489, loss = 0.001312
grad_step = 000490, loss = 0.001295
grad_step = 000491, loss = 0.001295
grad_step = 000492, loss = 0.001306
grad_step = 000493, loss = 0.001321
grad_step = 000494, loss = 0.001330
grad_step = 000495, loss = 0.001329
grad_step = 000496, loss = 0.001318
grad_step = 000497, loss = 0.001305
grad_step = 000498, loss = 0.001298
grad_step = 000499, loss = 0.001299
grad_step = 000500, loss = 0.001308
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001317
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

  date_run                              2020-05-11 06:41:12.878402
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.280105
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 06:41:12.884885
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.197244
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 06:41:12.893357
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.159901
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 06:41:12.900333
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.99719
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
0   2020-05-11 06:40:39.756074  ...    mean_absolute_error
1   2020-05-11 06:40:39.760585  ...     mean_squared_error
2   2020-05-11 06:40:39.765100  ...  median_absolute_error
3   2020-05-11 06:40:39.768710  ...               r2_score
4   2020-05-11 06:40:49.285253  ...    mean_absolute_error
5   2020-05-11 06:40:49.289729  ...     mean_squared_error
6   2020-05-11 06:40:49.293909  ...  median_absolute_error
7   2020-05-11 06:40:49.297715  ...               r2_score
8   2020-05-11 06:41:12.878402  ...    mean_absolute_error
9   2020-05-11 06:41:12.884885  ...     mean_squared_error
10  2020-05-11 06:41:12.893357  ...  median_absolute_error
11  2020-05-11 06:41:12.900333  ...               r2_score

[12 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 84, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 89, in module_load
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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140099402072472
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140098122803968
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140098122804472
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140098122804976
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140098122907944
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140098122908448

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6b6f188ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.530789
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.499481
grad_step = 000002, loss = 0.475365
grad_step = 000003, loss = 0.452792
grad_step = 000004, loss = 0.428190
grad_step = 000005, loss = 0.402092
grad_step = 000006, loss = 0.375840
grad_step = 000007, loss = 0.355180
grad_step = 000008, loss = 0.341347
grad_step = 000009, loss = 0.322708
grad_step = 000010, loss = 0.295893
grad_step = 000011, loss = 0.271866
grad_step = 000012, loss = 0.256868
grad_step = 000013, loss = 0.247844
grad_step = 000014, loss = 0.233584
grad_step = 000015, loss = 0.217493
grad_step = 000016, loss = 0.208103
grad_step = 000017, loss = 0.198297
grad_step = 000018, loss = 0.187716
grad_step = 000019, loss = 0.177548
grad_step = 000020, loss = 0.166244
grad_step = 000021, loss = 0.156055
grad_step = 000022, loss = 0.147894
grad_step = 000023, loss = 0.139076
grad_step = 000024, loss = 0.128902
grad_step = 000025, loss = 0.119881
grad_step = 000026, loss = 0.112935
grad_step = 000027, loss = 0.105841
grad_step = 000028, loss = 0.098630
grad_step = 000029, loss = 0.091888
grad_step = 000030, loss = 0.085136
grad_step = 000031, loss = 0.079235
grad_step = 000032, loss = 0.074018
grad_step = 000033, loss = 0.068751
grad_step = 000034, loss = 0.063803
grad_step = 000035, loss = 0.059365
grad_step = 000036, loss = 0.054844
grad_step = 000037, loss = 0.050497
grad_step = 000038, loss = 0.047128
grad_step = 000039, loss = 0.043982
grad_step = 000040, loss = 0.040609
grad_step = 000041, loss = 0.037680
grad_step = 000042, loss = 0.034954
grad_step = 000043, loss = 0.032310
grad_step = 000044, loss = 0.029928
grad_step = 000045, loss = 0.027670
grad_step = 000046, loss = 0.025489
grad_step = 000047, loss = 0.023546
grad_step = 000048, loss = 0.021638
grad_step = 000049, loss = 0.019951
grad_step = 000050, loss = 0.018483
grad_step = 000051, loss = 0.017005
grad_step = 000052, loss = 0.015776
grad_step = 000053, loss = 0.014718
grad_step = 000054, loss = 0.013597
grad_step = 000055, loss = 0.012592
grad_step = 000056, loss = 0.011666
grad_step = 000057, loss = 0.010740
grad_step = 000058, loss = 0.009966
grad_step = 000059, loss = 0.009257
grad_step = 000060, loss = 0.008600
grad_step = 000061, loss = 0.008042
grad_step = 000062, loss = 0.007512
grad_step = 000063, loss = 0.007027
grad_step = 000064, loss = 0.006574
grad_step = 000065, loss = 0.006153
grad_step = 000066, loss = 0.005792
grad_step = 000067, loss = 0.005438
grad_step = 000068, loss = 0.005120
grad_step = 000069, loss = 0.004853
grad_step = 000070, loss = 0.004591
grad_step = 000071, loss = 0.004366
grad_step = 000072, loss = 0.004154
grad_step = 000073, loss = 0.003950
grad_step = 000074, loss = 0.003780
grad_step = 000075, loss = 0.003612
grad_step = 000076, loss = 0.003459
grad_step = 000077, loss = 0.003330
grad_step = 000078, loss = 0.003211
grad_step = 000079, loss = 0.003108
grad_step = 000080, loss = 0.003006
grad_step = 000081, loss = 0.002912
grad_step = 000082, loss = 0.002832
grad_step = 000083, loss = 0.002762
grad_step = 000084, loss = 0.002702
grad_step = 000085, loss = 0.002646
grad_step = 000086, loss = 0.002595
grad_step = 000087, loss = 0.002548
grad_step = 000088, loss = 0.002506
grad_step = 000089, loss = 0.002471
grad_step = 000090, loss = 0.002439
grad_step = 000091, loss = 0.002415
grad_step = 000092, loss = 0.002392
grad_step = 000093, loss = 0.002371
grad_step = 000094, loss = 0.002353
grad_step = 000095, loss = 0.002335
grad_step = 000096, loss = 0.002322
grad_step = 000097, loss = 0.002310
grad_step = 000098, loss = 0.002301
grad_step = 000099, loss = 0.002292
grad_step = 000100, loss = 0.002285
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002279
grad_step = 000102, loss = 0.002273
grad_step = 000103, loss = 0.002268
grad_step = 000104, loss = 0.002264
grad_step = 000105, loss = 0.002260
grad_step = 000106, loss = 0.002257
grad_step = 000107, loss = 0.002255
grad_step = 000108, loss = 0.002252
grad_step = 000109, loss = 0.002250
grad_step = 000110, loss = 0.002247
grad_step = 000111, loss = 0.002245
grad_step = 000112, loss = 0.002243
grad_step = 000113, loss = 0.002241
grad_step = 000114, loss = 0.002239
grad_step = 000115, loss = 0.002236
grad_step = 000116, loss = 0.002234
grad_step = 000117, loss = 0.002232
grad_step = 000118, loss = 0.002230
grad_step = 000119, loss = 0.002227
grad_step = 000120, loss = 0.002225
grad_step = 000121, loss = 0.002222
grad_step = 000122, loss = 0.002220
grad_step = 000123, loss = 0.002217
grad_step = 000124, loss = 0.002214
grad_step = 000125, loss = 0.002211
grad_step = 000126, loss = 0.002209
grad_step = 000127, loss = 0.002206
grad_step = 000128, loss = 0.002203
grad_step = 000129, loss = 0.002200
grad_step = 000130, loss = 0.002198
grad_step = 000131, loss = 0.002195
grad_step = 000132, loss = 0.002192
grad_step = 000133, loss = 0.002189
grad_step = 000134, loss = 0.002186
grad_step = 000135, loss = 0.002184
grad_step = 000136, loss = 0.002181
grad_step = 000137, loss = 0.002178
grad_step = 000138, loss = 0.002175
grad_step = 000139, loss = 0.002172
grad_step = 000140, loss = 0.002170
grad_step = 000141, loss = 0.002167
grad_step = 000142, loss = 0.002164
grad_step = 000143, loss = 0.002161
grad_step = 000144, loss = 0.002159
grad_step = 000145, loss = 0.002156
grad_step = 000146, loss = 0.002153
grad_step = 000147, loss = 0.002150
grad_step = 000148, loss = 0.002148
grad_step = 000149, loss = 0.002145
grad_step = 000150, loss = 0.002142
grad_step = 000151, loss = 0.002139
grad_step = 000152, loss = 0.002136
grad_step = 000153, loss = 0.002133
grad_step = 000154, loss = 0.002130
grad_step = 000155, loss = 0.002127
grad_step = 000156, loss = 0.002125
grad_step = 000157, loss = 0.002122
grad_step = 000158, loss = 0.002118
grad_step = 000159, loss = 0.002115
grad_step = 000160, loss = 0.002112
grad_step = 000161, loss = 0.002109
grad_step = 000162, loss = 0.002106
grad_step = 000163, loss = 0.002103
grad_step = 000164, loss = 0.002100
grad_step = 000165, loss = 0.002096
grad_step = 000166, loss = 0.002093
grad_step = 000167, loss = 0.002090
grad_step = 000168, loss = 0.002086
grad_step = 000169, loss = 0.002083
grad_step = 000170, loss = 0.002079
grad_step = 000171, loss = 0.002076
grad_step = 000172, loss = 0.002072
grad_step = 000173, loss = 0.002069
grad_step = 000174, loss = 0.002065
grad_step = 000175, loss = 0.002061
grad_step = 000176, loss = 0.002058
grad_step = 000177, loss = 0.002054
grad_step = 000178, loss = 0.002050
grad_step = 000179, loss = 0.002046
grad_step = 000180, loss = 0.002042
grad_step = 000181, loss = 0.002039
grad_step = 000182, loss = 0.002035
grad_step = 000183, loss = 0.002031
grad_step = 000184, loss = 0.002026
grad_step = 000185, loss = 0.002023
grad_step = 000186, loss = 0.002018
grad_step = 000187, loss = 0.002014
grad_step = 000188, loss = 0.002010
grad_step = 000189, loss = 0.002006
grad_step = 000190, loss = 0.002002
grad_step = 000191, loss = 0.001998
grad_step = 000192, loss = 0.001995
grad_step = 000193, loss = 0.001993
grad_step = 000194, loss = 0.001987
grad_step = 000195, loss = 0.001981
grad_step = 000196, loss = 0.001977
grad_step = 000197, loss = 0.001974
grad_step = 000198, loss = 0.001970
grad_step = 000199, loss = 0.001964
grad_step = 000200, loss = 0.001960
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001957
grad_step = 000202, loss = 0.001953
grad_step = 000203, loss = 0.001947
grad_step = 000204, loss = 0.001942
grad_step = 000205, loss = 0.001939
grad_step = 000206, loss = 0.001935
grad_step = 000207, loss = 0.001931
grad_step = 000208, loss = 0.001925
grad_step = 000209, loss = 0.001921
grad_step = 000210, loss = 0.001917
grad_step = 000211, loss = 0.001913
grad_step = 000212, loss = 0.001909
grad_step = 000213, loss = 0.001904
grad_step = 000214, loss = 0.001900
grad_step = 000215, loss = 0.001897
grad_step = 000216, loss = 0.001895
grad_step = 000217, loss = 0.001893
grad_step = 000218, loss = 0.001892
grad_step = 000219, loss = 0.001889
grad_step = 000220, loss = 0.001885
grad_step = 000221, loss = 0.001878
grad_step = 000222, loss = 0.001872
grad_step = 000223, loss = 0.001869
grad_step = 000224, loss = 0.001868
grad_step = 000225, loss = 0.001862
grad_step = 000226, loss = 0.001854
grad_step = 000227, loss = 0.001851
grad_step = 000228, loss = 0.001853
grad_step = 000229, loss = 0.001852
grad_step = 000230, loss = 0.001843
grad_step = 000231, loss = 0.001837
grad_step = 000232, loss = 0.001836
grad_step = 000233, loss = 0.001834
grad_step = 000234, loss = 0.001826
grad_step = 000235, loss = 0.001820
grad_step = 000236, loss = 0.001817
grad_step = 000237, loss = 0.001816
grad_step = 000238, loss = 0.001813
grad_step = 000239, loss = 0.001809
grad_step = 000240, loss = 0.001806
grad_step = 000241, loss = 0.001810
grad_step = 000242, loss = 0.001819
grad_step = 000243, loss = 0.001835
grad_step = 000244, loss = 0.001845
grad_step = 000245, loss = 0.001849
grad_step = 000246, loss = 0.001824
grad_step = 000247, loss = 0.001796
grad_step = 000248, loss = 0.001778
grad_step = 000249, loss = 0.001785
grad_step = 000250, loss = 0.001803
grad_step = 000251, loss = 0.001800
grad_step = 000252, loss = 0.001781
grad_step = 000253, loss = 0.001759
grad_step = 000254, loss = 0.001758
grad_step = 000255, loss = 0.001768
grad_step = 000256, loss = 0.001770
grad_step = 000257, loss = 0.001776
grad_step = 000258, loss = 0.001772
grad_step = 000259, loss = 0.001761
grad_step = 000260, loss = 0.001744
grad_step = 000261, loss = 0.001734
grad_step = 000262, loss = 0.001736
grad_step = 000263, loss = 0.001741
grad_step = 000264, loss = 0.001744
grad_step = 000265, loss = 0.001743
grad_step = 000266, loss = 0.001746
grad_step = 000267, loss = 0.001744
grad_step = 000268, loss = 0.001740
grad_step = 000269, loss = 0.001731
grad_step = 000270, loss = 0.001723
grad_step = 000271, loss = 0.001717
grad_step = 000272, loss = 0.001713
grad_step = 000273, loss = 0.001709
grad_step = 000274, loss = 0.001705
grad_step = 000275, loss = 0.001702
grad_step = 000276, loss = 0.001701
grad_step = 000277, loss = 0.001700
grad_step = 000278, loss = 0.001698
grad_step = 000279, loss = 0.001697
grad_step = 000280, loss = 0.001699
grad_step = 000281, loss = 0.001711
grad_step = 000282, loss = 0.001744
grad_step = 000283, loss = 0.001839
grad_step = 000284, loss = 0.001931
grad_step = 000285, loss = 0.002024
grad_step = 000286, loss = 0.001808
grad_step = 000287, loss = 0.001697
grad_step = 000288, loss = 0.001789
grad_step = 000289, loss = 0.001806
grad_step = 000290, loss = 0.001715
grad_step = 000291, loss = 0.001714
grad_step = 000292, loss = 0.001768
grad_step = 000293, loss = 0.001729
grad_step = 000294, loss = 0.001682
grad_step = 000295, loss = 0.001733
grad_step = 000296, loss = 0.001760
grad_step = 000297, loss = 0.001698
grad_step = 000298, loss = 0.001681
grad_step = 000299, loss = 0.001715
grad_step = 000300, loss = 0.001721
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001688
grad_step = 000302, loss = 0.001674
grad_step = 000303, loss = 0.001693
grad_step = 000304, loss = 0.001697
grad_step = 000305, loss = 0.001679
grad_step = 000306, loss = 0.001674
grad_step = 000307, loss = 0.001679
grad_step = 000308, loss = 0.001676
grad_step = 000309, loss = 0.001667
grad_step = 000310, loss = 0.001666
grad_step = 000311, loss = 0.001671
grad_step = 000312, loss = 0.001665
grad_step = 000313, loss = 0.001655
grad_step = 000314, loss = 0.001658
grad_step = 000315, loss = 0.001662
grad_step = 000316, loss = 0.001657
grad_step = 000317, loss = 0.001650
grad_step = 000318, loss = 0.001647
grad_step = 000319, loss = 0.001651
grad_step = 000320, loss = 0.001652
grad_step = 000321, loss = 0.001645
grad_step = 000322, loss = 0.001640
grad_step = 000323, loss = 0.001641
grad_step = 000324, loss = 0.001642
grad_step = 000325, loss = 0.001641
grad_step = 000326, loss = 0.001637
grad_step = 000327, loss = 0.001633
grad_step = 000328, loss = 0.001633
grad_step = 000329, loss = 0.001634
grad_step = 000330, loss = 0.001632
grad_step = 000331, loss = 0.001629
grad_step = 000332, loss = 0.001626
grad_step = 000333, loss = 0.001625
grad_step = 000334, loss = 0.001625
grad_step = 000335, loss = 0.001624
grad_step = 000336, loss = 0.001622
grad_step = 000337, loss = 0.001620
grad_step = 000338, loss = 0.001618
grad_step = 000339, loss = 0.001616
grad_step = 000340, loss = 0.001615
grad_step = 000341, loss = 0.001614
grad_step = 000342, loss = 0.001613
grad_step = 000343, loss = 0.001611
grad_step = 000344, loss = 0.001609
grad_step = 000345, loss = 0.001608
grad_step = 000346, loss = 0.001606
grad_step = 000347, loss = 0.001605
grad_step = 000348, loss = 0.001605
grad_step = 000349, loss = 0.001605
grad_step = 000350, loss = 0.001607
grad_step = 000351, loss = 0.001612
grad_step = 000352, loss = 0.001623
grad_step = 000353, loss = 0.001646
grad_step = 000354, loss = 0.001681
grad_step = 000355, loss = 0.001744
grad_step = 000356, loss = 0.001769
grad_step = 000357, loss = 0.001778
grad_step = 000358, loss = 0.001682
grad_step = 000359, loss = 0.001609
grad_step = 000360, loss = 0.001616
grad_step = 000361, loss = 0.001658
grad_step = 000362, loss = 0.001663
grad_step = 000363, loss = 0.001613
grad_step = 000364, loss = 0.001593
grad_step = 000365, loss = 0.001620
grad_step = 000366, loss = 0.001653
grad_step = 000367, loss = 0.001667
grad_step = 000368, loss = 0.001635
grad_step = 000369, loss = 0.001593
grad_step = 000370, loss = 0.001574
grad_step = 000371, loss = 0.001587
grad_step = 000372, loss = 0.001611
grad_step = 000373, loss = 0.001616
grad_step = 000374, loss = 0.001601
grad_step = 000375, loss = 0.001576
grad_step = 000376, loss = 0.001565
grad_step = 000377, loss = 0.001572
grad_step = 000378, loss = 0.001584
grad_step = 000379, loss = 0.001588
grad_step = 000380, loss = 0.001578
grad_step = 000381, loss = 0.001565
grad_step = 000382, loss = 0.001558
grad_step = 000383, loss = 0.001559
grad_step = 000384, loss = 0.001565
grad_step = 000385, loss = 0.001568
grad_step = 000386, loss = 0.001567
grad_step = 000387, loss = 0.001561
grad_step = 000388, loss = 0.001553
grad_step = 000389, loss = 0.001549
grad_step = 000390, loss = 0.001547
grad_step = 000391, loss = 0.001549
grad_step = 000392, loss = 0.001551
grad_step = 000393, loss = 0.001552
grad_step = 000394, loss = 0.001552
grad_step = 000395, loss = 0.001550
grad_step = 000396, loss = 0.001547
grad_step = 000397, loss = 0.001544
grad_step = 000398, loss = 0.001541
grad_step = 000399, loss = 0.001539
grad_step = 000400, loss = 0.001538
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001539
grad_step = 000402, loss = 0.001543
grad_step = 000403, loss = 0.001553
grad_step = 000404, loss = 0.001580
grad_step = 000405, loss = 0.001633
grad_step = 000406, loss = 0.001747
grad_step = 000407, loss = 0.001877
grad_step = 000408, loss = 0.002030
grad_step = 000409, loss = 0.001874
grad_step = 000410, loss = 0.001642
grad_step = 000411, loss = 0.001545
grad_step = 000412, loss = 0.001677
grad_step = 000413, loss = 0.001772
grad_step = 000414, loss = 0.001608
grad_step = 000415, loss = 0.001541
grad_step = 000416, loss = 0.001649
grad_step = 000417, loss = 0.001636
grad_step = 000418, loss = 0.001536
grad_step = 000419, loss = 0.001555
grad_step = 000420, loss = 0.001609
grad_step = 000421, loss = 0.001562
grad_step = 000422, loss = 0.001518
grad_step = 000423, loss = 0.001565
grad_step = 000424, loss = 0.001579
grad_step = 000425, loss = 0.001525
grad_step = 000426, loss = 0.001526
grad_step = 000427, loss = 0.001563
grad_step = 000428, loss = 0.001541
grad_step = 000429, loss = 0.001514
grad_step = 000430, loss = 0.001533
grad_step = 000431, loss = 0.001549
grad_step = 000432, loss = 0.001533
grad_step = 000433, loss = 0.001530
grad_step = 000434, loss = 0.001560
grad_step = 000435, loss = 0.001586
grad_step = 000436, loss = 0.001603
grad_step = 000437, loss = 0.001645
grad_step = 000438, loss = 0.001721
grad_step = 000439, loss = 0.001748
grad_step = 000440, loss = 0.001748
grad_step = 000441, loss = 0.001686
grad_step = 000442, loss = 0.001605
grad_step = 000443, loss = 0.001527
grad_step = 000444, loss = 0.001498
grad_step = 000445, loss = 0.001536
grad_step = 000446, loss = 0.001591
grad_step = 000447, loss = 0.001600
grad_step = 000448, loss = 0.001554
grad_step = 000449, loss = 0.001505
grad_step = 000450, loss = 0.001491
grad_step = 000451, loss = 0.001512
grad_step = 000452, loss = 0.001536
grad_step = 000453, loss = 0.001540
grad_step = 000454, loss = 0.001521
grad_step = 000455, loss = 0.001494
grad_step = 000456, loss = 0.001482
grad_step = 000457, loss = 0.001489
grad_step = 000458, loss = 0.001504
grad_step = 000459, loss = 0.001512
grad_step = 000460, loss = 0.001503
grad_step = 000461, loss = 0.001487
grad_step = 000462, loss = 0.001475
grad_step = 000463, loss = 0.001474
grad_step = 000464, loss = 0.001481
grad_step = 000465, loss = 0.001488
grad_step = 000466, loss = 0.001489
grad_step = 000467, loss = 0.001484
grad_step = 000468, loss = 0.001476
grad_step = 000469, loss = 0.001469
grad_step = 000470, loss = 0.001466
grad_step = 000471, loss = 0.001466
grad_step = 000472, loss = 0.001468
grad_step = 000473, loss = 0.001471
grad_step = 000474, loss = 0.001472
grad_step = 000475, loss = 0.001471
grad_step = 000476, loss = 0.001468
grad_step = 000477, loss = 0.001465
grad_step = 000478, loss = 0.001461
grad_step = 000479, loss = 0.001458
grad_step = 000480, loss = 0.001455
grad_step = 000481, loss = 0.001454
grad_step = 000482, loss = 0.001453
grad_step = 000483, loss = 0.001452
grad_step = 000484, loss = 0.001452
grad_step = 000485, loss = 0.001452
grad_step = 000486, loss = 0.001452
grad_step = 000487, loss = 0.001453
grad_step = 000488, loss = 0.001455
grad_step = 000489, loss = 0.001458
grad_step = 000490, loss = 0.001463
grad_step = 000491, loss = 0.001470
grad_step = 000492, loss = 0.001483
grad_step = 000493, loss = 0.001500
grad_step = 000494, loss = 0.001530
grad_step = 000495, loss = 0.001560
grad_step = 000496, loss = 0.001606
grad_step = 000497, loss = 0.001627
grad_step = 000498, loss = 0.001640
grad_step = 000499, loss = 0.001602
grad_step = 000500, loss = 0.001544
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001482
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

  date_run                              2020-05-11 06:41:40.405512
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.269421
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 06:41:40.411945
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.182314
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 06:41:40.422267
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.156584
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 06:41:40.428959
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.77033
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6b6f1801d0> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355939.1250
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 252883.2656
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 169559.0312
Epoch 4/10

1/1 [==============================] - 0s 110ms/step - loss: 99877.4219
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 54741.3320
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 30804.3633
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 18908.6074
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 12570.7715
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 8904.7930
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 6663.1040

  #### Inference Need return ypred, ytrue ######################### 
[[-7.85628974e-01  3.63167048e-01 -1.39710081e+00  1.71303236e+00
  -4.54741895e-01 -1.19661689e+00 -1.00463808e-01 -3.96032810e-01
   5.99874556e-02 -6.48386002e-01 -1.61907506e+00  7.41853237e-01
   2.20288038e+00  7.54204571e-01 -7.48454332e-02 -4.30420578e-01
  -6.93571687e-01  3.56140733e-02  1.49993718e-01 -1.16246724e+00
  -1.05261135e+00  7.99656510e-01 -1.60333610e+00 -1.64886570e+00
  -5.75163186e-01 -1.55097270e+00 -1.46261322e+00 -4.60776031e-01
  -4.90326524e-01  8.79476666e-01  4.01995778e-01  5.12848258e-01
   1.96089125e+00  9.24011588e-01  2.92874455e-01 -1.86377120e+00
   1.28125393e+00 -2.13034010e+00  1.15972829e+00 -3.85943294e-01
  -1.69948131e-01 -4.85986233e-01  8.42079699e-01 -4.16565061e-01
   5.90437770e-01 -3.52371991e-01 -8.61413896e-01  1.05352008e+00
  -5.32004178e-01 -1.36527145e+00  3.29727441e-01  8.93628001e-01
   2.60187536e-02  9.61095631e-01  5.47975063e-01  1.16509426e+00
   8.10825229e-02  1.23149729e+00  1.62338987e-01 -9.60967720e-01
  -3.38137969e-02  8.75639534e+00  8.44457054e+00  7.74093103e+00
   7.49296045e+00  7.27447796e+00  8.50808048e+00  9.65416431e+00
   7.74078321e+00  7.70233297e+00  9.19578266e+00  8.70383263e+00
   8.47820759e+00  9.05014992e+00  7.72285748e+00  8.23885918e+00
   6.47099066e+00  9.82094860e+00  8.05006123e+00  9.03319168e+00
   8.16881657e+00  9.42981815e+00  8.43300819e+00  8.22501945e+00
   8.54654598e+00  7.16761303e+00  6.02081585e+00  8.48793697e+00
   6.47406340e+00  6.46312571e+00  8.22244358e+00  9.51487350e+00
   9.18134785e+00  7.53561735e+00  7.23819780e+00  8.36784554e+00
   8.42468739e+00  6.14964151e+00  7.35938644e+00  8.12174034e+00
   9.53114319e+00  8.73026848e+00  6.97598839e+00  9.76499844e+00
   5.36648941e+00  7.34851885e+00  7.22180843e+00  9.76765347e+00
   7.99198723e+00  8.07289886e+00  9.86293793e+00  6.25710440e+00
   8.13970757e+00  9.67931175e+00  8.75697708e+00  8.89839935e+00
   8.49650955e+00  9.95240593e+00  8.76125622e+00  8.15689850e+00
  -9.45847809e-01  1.71584457e-01 -5.40879726e-01 -1.21208274e+00
   8.28916013e-01 -1.08230996e+00 -1.21472883e+00 -2.31240407e-01
   6.06841922e-01  6.87901974e-01  7.97467232e-01 -5.17746925e-01
  -4.32512313e-01  1.52317166e+00  5.95083773e-01 -2.36647177e+00
  -1.14243329e-01 -7.83097863e-01 -1.79334307e+00 -2.01564145e+00
  -1.40233183e+00  1.72972524e+00 -6.67558670e-01 -4.04160023e-02
  -5.27574182e-01 -1.08701634e+00  3.11267197e-01 -1.06460857e+00
  -1.12388730e-02  1.98580861e+00 -4.22638535e-01 -1.13944376e+00
   1.18371415e+00  1.90756059e+00  1.17851126e+00 -5.48498929e-01
  -4.59053129e-01  1.22980320e+00 -6.62428141e-02  1.03050148e+00
  -1.44891441e-01  2.46226057e-01 -1.65703082e+00  8.78013074e-01
  -1.30513966e+00 -5.84989607e-01 -1.05589435e-01  1.31333992e-03
   1.16925311e+00  4.13773060e-01  1.02557755e+00 -2.53340513e-01
  -4.80579734e-02 -2.69087553e-02 -1.83278775e+00  3.44573140e-01
   3.66501182e-01  5.78846216e-01  8.57473791e-01  1.40441537e+00
   3.17100406e-01  1.72506094e+00  1.02227259e+00  2.57303095e+00
   2.05420256e+00  7.84075439e-01  1.13509691e+00  8.27000082e-01
   2.26784945e-01  1.43044043e+00  1.86381817e-01  1.85132504e+00
   1.55052316e+00  2.21493363e-01  5.76776683e-01  2.05325007e-01
   3.00277281e+00  8.32728207e-01  1.40208960e+00  4.69018579e-01
   4.77559805e-01  1.82095146e+00  1.01746297e+00  4.72389638e-01
   2.19079792e-01  1.10248530e+00  2.31477928e+00  7.57703424e-01
   7.06872344e-01  1.33547068e+00  5.62919736e-01  1.94768429e+00
   4.66954052e-01  3.66784453e-01  1.12936759e+00  2.90008664e-01
   8.61541033e-01  2.34753895e+00  1.25971401e+00  2.32058644e-01
   1.04751873e+00  1.35571122e+00  1.05783439e+00  7.32969463e-01
   7.91338563e-01  3.04774761e+00  1.64443946e+00  2.05025101e+00
   2.52570868e-01  2.96640158e-01  4.56681371e-01  9.28270221e-01
   1.61597538e+00  2.46804762e+00  8.53252053e-01  1.66477406e+00
   1.28798163e+00  8.11947286e-01  2.03105021e+00  2.49616742e-01
   1.59376144e-01  8.08435535e+00  8.62303257e+00  8.12842083e+00
   6.79633856e+00  8.42886448e+00  9.12662411e+00  8.84828091e+00
   8.42582130e+00  1.11996002e+01  9.27825832e+00  7.65771151e+00
   9.77265167e+00  9.56034184e+00  8.08177948e+00  7.62481546e+00
   6.83839703e+00  8.30565071e+00  9.99769211e+00  8.32601738e+00
   7.30330706e+00  8.89170742e+00  7.29255676e+00  8.99716282e+00
   9.53956985e+00  7.69052935e+00  8.64924431e+00  9.44042778e+00
   8.66169453e+00  8.38767815e+00  9.91036510e+00  8.03580856e+00
   8.61770916e+00  9.40036869e+00  9.01587677e+00  9.06227493e+00
   9.13359547e+00  9.34676361e+00  7.73429394e+00  8.71088314e+00
   6.67651367e+00  1.01952753e+01  7.67789841e+00  8.77848244e+00
   8.51761055e+00  9.25074577e+00  8.02012348e+00  8.82846928e+00
   6.69306040e+00  9.74886322e+00  8.62357807e+00  7.18342257e+00
   7.26314688e+00  9.64796352e+00  8.27002430e+00  7.32648087e+00
   8.43269062e+00  8.29750538e+00  8.89393711e+00  7.94685984e+00
   1.93398714e-01  8.26722741e-01  1.79706335e+00  4.65596557e-01
   1.95134687e+00  1.97477818e+00  2.12808847e-01  1.90449953e+00
   1.54995263e+00  1.32165456e+00  2.68499565e+00  2.08148503e+00
   1.78182626e+00  1.46687841e+00  5.79456329e-01  2.44088292e-01
   2.60410500e+00  3.58342588e-01  2.07226205e+00  3.26739788e-01
   4.86537933e-01  4.52848792e-01  1.84788346e+00  2.24563050e+00
   2.76652575e-01  2.82266617e+00  9.35176075e-01  1.12616277e+00
   6.56187832e-01  1.68891931e+00  1.76139784e+00  1.62265766e+00
   1.64466012e+00  3.13993406e+00  2.30463529e+00  6.51667833e-01
   1.53395820e+00  3.39347959e-01  4.82037246e-01  1.90448284e+00
   1.84929848e+00  2.65932739e-01  1.25110579e+00  6.04224682e-01
   3.11321640e+00  1.09155977e+00  1.90548360e+00  1.36610734e+00
   3.13348949e-01  2.51026011e+00  1.91532779e+00  5.56762040e-01
   1.71876585e+00  4.68344331e-01  3.93212676e-01  2.11501002e+00
   1.13726699e+00  3.60766709e-01  1.34461832e+00  5.88117778e-01
  -1.04325914e+01  3.69263554e+00 -7.16557741e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:41:50.033645
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   93.2053
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 06:41:50.038169
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8706.95
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 06:41:50.041949
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   93.4356
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 06:41:50.045769
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -778.769
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6ae4a103c8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:42:08.002666
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 06:42:08.006996
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 06:42:08.010847
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 06:42:08.014987
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
0   2020-05-11 06:41:40.405512  ...    mean_absolute_error
1   2020-05-11 06:41:40.411945  ...     mean_squared_error
2   2020-05-11 06:41:40.422267  ...  median_absolute_error
3   2020-05-11 06:41:40.428959  ...               r2_score
4   2020-05-11 06:41:50.033645  ...    mean_absolute_error
5   2020-05-11 06:41:50.038169  ...     mean_squared_error
6   2020-05-11 06:41:50.041949  ...  median_absolute_error
7   2020-05-11 06:41:50.045769  ...               r2_score
8   2020-05-11 06:42:08.002666  ...    mean_absolute_error
9   2020-05-11 06:42:08.006996  ...     mean_squared_error
10  2020-05-11 06:42:08.010847  ...  median_absolute_error
11  2020-05-11 06:42:08.014987  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 118, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

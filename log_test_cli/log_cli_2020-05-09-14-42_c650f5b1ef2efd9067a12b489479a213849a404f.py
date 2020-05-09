
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_cli GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_cli 

  # Testing Command Line System   





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'c650f5b1ef2efd9067a12b489479a213849a404f', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c650f5b1ef2efd9067a12b489479a213849a404f

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n']





 ************************************************************************************************************************
ml_models --do init  --path ztest/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
init

  Working Folder ztest/ 

  Config values {'model_trained': 'ztest//model_trained/', 'dataset': 'ztest//dataset/'} 
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 483, in main
    config_init(to_path=arg.path)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 380, in config_init
    log("Config path", get_pretrained_path())
NameError: name 'get_pretrained_path' is not defined





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
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 503, in main
    fit_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 407, in fit_cli
    model, sess = module.fit(model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 86, in fit
    df = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 232, in get_dataset
    df = pd.read_csv(filename)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timseries/GOOG-year.csv' does not exist: b'/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timseries/GOOG-year.csv'





 ************************************************************************************************************************
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
predict
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 507, in main
    predict_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 422, in predict_cli
    model, session = load(load_pars)
TypeError: load() missing 1 required positional argument: 'load_pars'





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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fdc1c7d97b8> 

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
 [-0.02788376 -0.02220443 -0.01620785 -0.01616212  0.12057253 -0.00809067]
 [ 0.07059845 -0.04417738  0.11145175 -0.03988602 -0.12323309  0.17420253]
 [ 0.00327247  0.17833212  0.14375514  0.00534554  0.02396636  0.03401732]
 [ 0.1323854   0.11104172 -0.3021957   0.18135525  0.29045454 -0.04494506]
 [-0.13907766 -0.31796354  0.18925466 -0.16438162  0.27313161 -0.67712605]
 [-0.02646243 -0.01711317  0.06861495  0.40775394  0.1839685  -0.34395656]
 [-0.2721526  -0.14218892  0.19286726 -0.2177441   0.49681181  0.15178227]
 [-0.08988906  0.51433772 -0.04886628  0.10603044 -0.36601183  0.17880067]
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
{'loss': 0.5150385797023773, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
{'loss': 0.36744123697280884, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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

  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f15f6c12828> 

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
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 8.67116228e-02 -8.41709040e-03 -6.11822978e-02  2.67023910e-02
  -1.40380144e-01  2.37239338e-02]
 [ 4.86610085e-02 -1.99475706e-01  8.83878991e-02 -7.89735094e-02
  -4.05958388e-04 -7.63250291e-02]
 [ 8.05906877e-02  1.43035036e-02  1.25037879e-01 -1.05169393e-01
  -1.13231964e-01 -2.53499508e-01]
 [ 1.16459087e-01  1.92329600e-01 -8.46605897e-02  5.21878526e-02
   1.03788912e-01  6.31756783e-02]
 [ 4.86947417e-01  1.31382883e-01 -1.54535443e-01 -6.85251117e-01
   1.71458587e-01 -2.45757565e-01]
 [-1.52079180e-01  3.85848850e-01  5.19594967e-01 -2.48793468e-01
   2.78928220e-01 -3.04602608e-02]
 [-7.03255683e-02 -1.43352509e-01  1.21255800e-01 -8.38963464e-02
  -3.76843005e-01  2.78049886e-01]
 [ 2.80585617e-01 -5.77870943e-02  2.67952323e-01  4.21928406e-01
   7.04116607e-03  1.88573375e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]]

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
{'loss': 0.4067152440547943, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
{'loss': 0.475702740252018, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





 ************************************************************************************************************************
ml_optim --do search  --config_file optim_config.json  --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 387, in main
    optim_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 245, in optim_cli
    js = json.load(open(config_file, mode='r'))  # Config
FileNotFoundError: [Errno 2] No such file or directory: 'optim_config.json'





 ************************************************************************************************************************
ml_optim --do search  --config_file optim_config_prune.json   --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 387, in main
    optim_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 245, in optim_cli
    js = json.load(open(config_file, mode='r'))  # Config
FileNotFoundError: [Errno 2] No such file or directory: 'optim_config_prune.json'





 ************************************************************************************************************************
ml_optim --do test   --model_uri model_tf.1_lstm.py   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
Deprecaton set to False

  {'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} {'engine': 'optuna', 'method': 'prune', 'ntrials': 5} {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}} 

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  ###### Hyper-optimization through study   ################################## 

  check <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]} 
[32m[I 2020-05-09 14:42:52,208][0m Finished trial#0 resulted in value: 0.3609362542629242. Current best value is 0.3609362542629242 with parameters: {'learning_rate': 0.0061397133923274975, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-09 14:42:53,187][0m Finished trial#1 resulted in value: 0.30714716762304306. Current best value is 0.30714716762304306 with parameters: {'learning_rate': 0.00243474205905716, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f16ee8f5e48> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:43:08.690017
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 14:43:08.693219
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 14:43:08.696053
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 14:43:08.699053
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f16e528e400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 354157.5000
Epoch 2/10

1/1 [==============================] - 0s 88ms/step - loss: 280961.5000
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 189891.3750
Epoch 4/10

1/1 [==============================] - 0s 73ms/step - loss: 118000.9844
Epoch 5/10

1/1 [==============================] - 0s 79ms/step - loss: 68921.6172
Epoch 6/10

1/1 [==============================] - 0s 82ms/step - loss: 40897.7148
Epoch 7/10

1/1 [==============================] - 0s 78ms/step - loss: 25706.0566
Epoch 8/10

1/1 [==============================] - 0s 83ms/step - loss: 17190.9375
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 12190.8916
Epoch 10/10

1/1 [==============================] - 0s 82ms/step - loss: 9123.7490

  #### Inference Need return ypred, ytrue ######################### 
[[ 9.6007347e-01  1.2603822e+00 -1.4091793e+00  5.7883209e-01
  -6.0873348e-01 -5.7721883e-01  4.2436910e-01 -4.1128251e-01
  -1.4190481e+00  9.4511163e-01  3.4605831e-01  4.0900263e-01
  -3.3839685e-01  1.1048751e+00  7.2963828e-01  6.9558650e-01
  -2.5456303e-01 -8.1651467e-01 -3.8193020e-01  1.5780111e-01
   1.5965652e-01  8.4009850e-01  9.8008287e-01 -1.3373922e+00
  -3.5536706e-02  5.4222393e-01  4.3479368e-01 -1.4000714e+00
  -2.8384393e-01 -8.2401806e-01  1.3933736e-01  5.7480127e-01
  -1.4032378e+00 -5.6716734e-01  9.4442505e-01 -1.0366776e+00
  -2.7199787e-01 -7.8278232e-01 -1.3097783e+00  4.0903470e-01
  -1.0651542e+00  5.9777802e-01  1.6241482e-01 -5.8365488e-01
  -7.3782802e-02 -3.6313367e-01  5.9117079e-01  3.8102099e-01
  -3.0223006e-01 -5.6022549e-01  4.7277904e-01 -2.3751503e-01
  -8.1936538e-01 -4.2209384e-01 -1.7187230e-01  7.5261062e-01
  -7.2833884e-01  1.1217502e+00 -5.2556515e-01 -1.5842593e-01
  -2.9710186e-01  5.8498996e-01 -1.6849284e-01 -2.2713773e-01
  -4.1906384e-01 -1.1935666e+00  1.5718091e+00  2.4527419e-01
   3.3138013e-01 -8.5713494e-01 -7.6704079e-01 -1.2032744e+00
   6.3436294e-01 -5.0577521e-01  7.4863839e-01 -8.6650401e-03
   3.0244607e-01 -1.5949392e-01  2.9469934e-01 -1.1070151e+00
  -1.3706006e+00  1.1435941e+00  5.1697828e-02  7.6487285e-01
  -3.6953801e-01  1.5537775e+00  3.8204810e-01  4.3959352e-01
   1.2407222e+00 -9.8621327e-01  1.8494591e-01  8.3742130e-01
  -1.8276675e-01  7.6674479e-01  5.3969926e-01  1.0074649e+00
   1.6132486e-01 -7.4050057e-01  5.2057010e-01 -4.1835308e-01
  -1.7902406e-01 -9.4579160e-02  6.2411118e-01 -2.2455975e-01
  -1.0885131e+00 -3.0444670e-01  3.7273729e-01  9.1538793e-01
   3.7591064e-01  5.2748901e-01 -5.6695604e-01  4.1825119e-01
  -1.9912907e-01  6.4076638e-01 -4.6189457e-01 -4.9935409e-01
  -1.4787604e+00  2.7948208e-02  7.5807256e-01  2.1882433e-01
   1.0183677e-01  5.8474584e+00  7.4793224e+00  7.5024867e+00
   6.2173052e+00  6.4470134e+00  6.9828100e+00  8.7662449e+00
   7.3679552e+00  7.8156724e+00  8.8623362e+00  5.8181205e+00
   6.7363172e+00  6.7442703e+00  5.9241142e+00  6.9256124e+00
   6.7444730e+00  8.1846170e+00  7.4020185e+00  7.3968325e+00
   6.7247577e+00  7.7462449e+00  8.1455154e+00  7.6987777e+00
   5.7022700e+00  7.3822322e+00  6.9826641e+00  8.0727596e+00
   5.7843680e+00  6.5870132e+00  6.4069629e+00  6.4899263e+00
   7.3169661e+00  7.0703535e+00  7.8620286e+00  6.8470812e+00
   6.5930834e+00  5.7785296e+00  6.5026526e+00  6.9446540e+00
   7.2756314e+00  7.1124907e+00  6.5148249e+00  7.4232311e+00
   7.9521537e+00  6.1092105e+00  6.6556783e+00  5.7311916e+00
   6.2079582e+00  6.9504595e+00  7.8799429e+00  5.7633743e+00
   6.9047885e+00  7.6816416e+00  5.5160851e+00  6.0802178e+00
   7.2174730e+00  6.2566342e+00  6.9131088e+00  5.2236214e+00
   1.1478918e+00  1.5194870e+00  2.2148967e+00  6.4648700e-01
   1.1708140e+00  6.6341215e-01  4.9356723e-01  1.2249854e+00
   7.5572979e-01  5.2025455e-01  4.6248841e-01  2.5435412e-01
   7.4430102e-01  2.0000248e+00  4.7049677e-01  4.1811228e-01
   1.3312650e+00  1.5301087e+00  3.1598511e+00  4.0024412e-01
   6.9745672e-01  2.5251281e-01  6.9965482e-01  2.1314368e+00
   1.2813197e+00  1.9303453e-01  1.6007786e+00  1.6104100e+00
   1.2522382e+00  1.4721336e+00  2.8474605e-01  5.7294863e-01
   2.6561506e+00  1.2245884e+00  9.4332087e-01  1.2067721e+00
   3.7466693e-01  9.0310812e-01  7.0341873e-01  9.1493642e-01
   3.2849807e-01  1.1116710e+00  7.9323971e-01  1.2109838e+00
   1.9778111e+00  2.2419724e+00  1.5249176e+00  1.8695104e+00
   7.4089062e-01  2.1594033e+00  1.3404298e+00  8.5735810e-01
   2.1770892e+00  6.9542348e-01  4.2440397e-01  1.4711891e+00
   9.9855244e-01  1.9799236e+00  1.6554970e+00  6.3705856e-01
   8.0509865e-01  1.5151942e+00  1.9002969e+00  8.2788557e-01
   6.4291853e-01  4.9734688e-01  6.7801195e-01  6.1117238e-01
   3.7105292e-01  4.8676836e-01  1.1958395e+00  9.7522628e-01
   4.2417908e-01  1.0431347e+00  1.4365250e+00  2.7622447e+00
   3.9056128e-01  8.6744344e-01  3.6796486e-01  1.2346076e+00
   5.9650242e-01  5.6863213e-01  8.2785404e-01  1.8308598e+00
   4.2987537e-01  6.6238439e-01  5.6099236e-01  3.1813163e-01
   2.3517592e+00  1.9023106e+00  1.5996685e+00  2.3637307e-01
   1.5907991e+00  2.7538052e+00  8.0443382e-01  1.4866588e+00
   5.3538281e-01  2.5310731e-01  9.5420671e-01  6.8195164e-01
   2.3682511e-01  1.2810999e+00  3.2545316e-01  1.3459445e+00
   1.0376232e+00  7.3649442e-01  6.9412643e-01  1.5159873e+00
   1.8152745e+00  1.9522672e+00  1.7894382e+00  1.1946089e+00
   1.1933187e+00  1.4523509e+00  1.3064132e+00  5.3477830e-01
   1.2660379e+00  3.7975907e-01  7.1307695e-01  1.1368543e+00
   8.7061882e-02  7.2270808e+00  6.3824944e+00  7.7846847e+00
   7.6799335e+00  7.3210635e+00  6.8068590e+00  8.8167162e+00
   8.3245907e+00  6.7195544e+00  6.9324117e+00  7.2680044e+00
   8.5430708e+00  6.7806277e+00  7.7139211e+00  8.0024042e+00
   5.8142014e+00  8.1321793e+00  6.9401312e+00  7.0100083e+00
   6.6190128e+00  8.1474047e+00  7.9871159e+00  7.1482038e+00
   6.6121416e+00  7.4350853e+00  7.5338516e+00  6.6210732e+00
   7.5170212e+00  7.4987092e+00  7.7403860e+00  7.5526242e+00
   6.6199841e+00  7.3186069e+00  5.9991498e+00  6.4107265e+00
   7.8530631e+00  6.8285775e+00  7.3587503e+00  6.6192188e+00
   7.4632111e+00  5.8536115e+00  5.7498913e+00  7.2729692e+00
   6.5928230e+00  6.5378380e+00  5.7593369e+00  7.8189797e+00
   6.5470672e+00  7.2939529e+00  7.1456180e+00  6.8444085e+00
   7.5180006e+00  6.1384492e+00  7.8073201e+00  7.4319959e+00
   7.1960683e+00  7.3973565e+00  6.8532333e+00  6.2531543e+00
  -4.0675564e+00 -6.8239179e+00  8.3386002e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:43:15.879990
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.8266
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 14:43:15.883615
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9015.48
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 14:43:15.886896
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                     95.19
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 14:43:15.889627
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    -806.4
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139735777719688
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139734836581768
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139734836582272
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139734836582776
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139734836583280
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139734836583784

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f16da6380f0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.639119
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.603534
grad_step = 000002, loss = 0.580282
grad_step = 000003, loss = 0.556656
grad_step = 000004, loss = 0.531091
grad_step = 000005, loss = 0.507315
grad_step = 000006, loss = 0.484742
grad_step = 000007, loss = 0.464038
grad_step = 000008, loss = 0.443858
grad_step = 000009, loss = 0.424675
grad_step = 000010, loss = 0.407611
grad_step = 000011, loss = 0.389681
grad_step = 000012, loss = 0.369823
grad_step = 000013, loss = 0.351443
grad_step = 000014, loss = 0.338521
grad_step = 000015, loss = 0.327775
grad_step = 000016, loss = 0.313642
grad_step = 000017, loss = 0.299401
grad_step = 000018, loss = 0.287744
grad_step = 000019, loss = 0.276819
grad_step = 000020, loss = 0.263929
grad_step = 000021, loss = 0.249545
grad_step = 000022, loss = 0.236302
grad_step = 000023, loss = 0.225507
grad_step = 000024, loss = 0.215764
grad_step = 000025, loss = 0.205877
grad_step = 000026, loss = 0.196077
grad_step = 000027, loss = 0.186342
grad_step = 000028, loss = 0.176593
grad_step = 000029, loss = 0.166961
grad_step = 000030, loss = 0.157573
grad_step = 000031, loss = 0.148646
grad_step = 000032, loss = 0.140322
grad_step = 000033, loss = 0.132538
grad_step = 000034, loss = 0.125069
grad_step = 000035, loss = 0.117863
grad_step = 000036, loss = 0.111141
grad_step = 000037, loss = 0.104818
grad_step = 000038, loss = 0.098521
grad_step = 000039, loss = 0.092310
grad_step = 000040, loss = 0.086491
grad_step = 000041, loss = 0.080748
grad_step = 000042, loss = 0.075307
grad_step = 000043, loss = 0.070488
grad_step = 000044, loss = 0.066068
grad_step = 000045, loss = 0.061718
grad_step = 000046, loss = 0.057466
grad_step = 000047, loss = 0.053508
grad_step = 000048, loss = 0.049795
grad_step = 000049, loss = 0.046207
grad_step = 000050, loss = 0.042793
grad_step = 000051, loss = 0.039632
grad_step = 000052, loss = 0.036700
grad_step = 000053, loss = 0.033890
grad_step = 000054, loss = 0.031247
grad_step = 000055, loss = 0.028799
grad_step = 000056, loss = 0.026478
grad_step = 000057, loss = 0.024339
grad_step = 000058, loss = 0.022328
grad_step = 000059, loss = 0.020372
grad_step = 000060, loss = 0.018619
grad_step = 000061, loss = 0.017001
grad_step = 000062, loss = 0.015498
grad_step = 000063, loss = 0.014115
grad_step = 000064, loss = 0.012834
grad_step = 000065, loss = 0.011654
grad_step = 000066, loss = 0.010583
grad_step = 000067, loss = 0.009607
grad_step = 000068, loss = 0.008725
grad_step = 000069, loss = 0.007954
grad_step = 000070, loss = 0.007239
grad_step = 000071, loss = 0.006605
grad_step = 000072, loss = 0.006049
grad_step = 000073, loss = 0.005550
grad_step = 000074, loss = 0.005108
grad_step = 000075, loss = 0.004714
grad_step = 000076, loss = 0.004376
grad_step = 000077, loss = 0.004092
grad_step = 000078, loss = 0.003838
grad_step = 000079, loss = 0.003616
grad_step = 000080, loss = 0.003424
grad_step = 000081, loss = 0.003254
grad_step = 000082, loss = 0.003110
grad_step = 000083, loss = 0.002985
grad_step = 000084, loss = 0.002884
grad_step = 000085, loss = 0.002794
grad_step = 000086, loss = 0.002712
grad_step = 000087, loss = 0.002642
grad_step = 000088, loss = 0.002579
grad_step = 000089, loss = 0.002524
grad_step = 000090, loss = 0.002475
grad_step = 000091, loss = 0.002432
grad_step = 000092, loss = 0.002393
grad_step = 000093, loss = 0.002355
grad_step = 000094, loss = 0.002321
grad_step = 000095, loss = 0.002289
grad_step = 000096, loss = 0.002260
grad_step = 000097, loss = 0.002234
grad_step = 000098, loss = 0.002210
grad_step = 000099, loss = 0.002188
grad_step = 000100, loss = 0.002166
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002147
grad_step = 000102, loss = 0.002130
grad_step = 000103, loss = 0.002115
grad_step = 000104, loss = 0.002101
grad_step = 000105, loss = 0.002089
grad_step = 000106, loss = 0.002078
grad_step = 000107, loss = 0.002068
grad_step = 000108, loss = 0.002059
grad_step = 000109, loss = 0.002052
grad_step = 000110, loss = 0.002045
grad_step = 000111, loss = 0.002040
grad_step = 000112, loss = 0.002034
grad_step = 000113, loss = 0.002030
grad_step = 000114, loss = 0.002025
grad_step = 000115, loss = 0.002021
grad_step = 000116, loss = 0.002018
grad_step = 000117, loss = 0.002014
grad_step = 000118, loss = 0.002011
grad_step = 000119, loss = 0.002008
grad_step = 000120, loss = 0.002004
grad_step = 000121, loss = 0.002001
grad_step = 000122, loss = 0.001998
grad_step = 000123, loss = 0.001994
grad_step = 000124, loss = 0.001991
grad_step = 000125, loss = 0.001987
grad_step = 000126, loss = 0.001984
grad_step = 000127, loss = 0.001980
grad_step = 000128, loss = 0.001976
grad_step = 000129, loss = 0.001972
grad_step = 000130, loss = 0.001968
grad_step = 000131, loss = 0.001965
grad_step = 000132, loss = 0.001961
grad_step = 000133, loss = 0.001957
grad_step = 000134, loss = 0.001953
grad_step = 000135, loss = 0.001949
grad_step = 000136, loss = 0.001945
grad_step = 000137, loss = 0.001942
grad_step = 000138, loss = 0.001938
grad_step = 000139, loss = 0.001935
grad_step = 000140, loss = 0.001930
grad_step = 000141, loss = 0.001926
grad_step = 000142, loss = 0.001923
grad_step = 000143, loss = 0.001920
grad_step = 000144, loss = 0.001917
grad_step = 000145, loss = 0.001913
grad_step = 000146, loss = 0.001909
grad_step = 000147, loss = 0.001906
grad_step = 000148, loss = 0.001903
grad_step = 000149, loss = 0.001900
grad_step = 000150, loss = 0.001897
grad_step = 000151, loss = 0.001893
grad_step = 000152, loss = 0.001890
grad_step = 000153, loss = 0.001887
grad_step = 000154, loss = 0.001883
grad_step = 000155, loss = 0.001880
grad_step = 000156, loss = 0.001877
grad_step = 000157, loss = 0.001873
grad_step = 000158, loss = 0.001871
grad_step = 000159, loss = 0.001869
grad_step = 000160, loss = 0.001868
grad_step = 000161, loss = 0.001870
grad_step = 000162, loss = 0.001874
grad_step = 000163, loss = 0.001873
grad_step = 000164, loss = 0.001866
grad_step = 000165, loss = 0.001854
grad_step = 000166, loss = 0.001847
grad_step = 000167, loss = 0.001845
grad_step = 000168, loss = 0.001848
grad_step = 000169, loss = 0.001849
grad_step = 000170, loss = 0.001846
grad_step = 000171, loss = 0.001840
grad_step = 000172, loss = 0.001833
grad_step = 000173, loss = 0.001827
grad_step = 000174, loss = 0.001825
grad_step = 000175, loss = 0.001825
grad_step = 000176, loss = 0.001828
grad_step = 000177, loss = 0.001833
grad_step = 000178, loss = 0.001842
grad_step = 000179, loss = 0.001848
grad_step = 000180, loss = 0.001849
grad_step = 000181, loss = 0.001835
grad_step = 000182, loss = 0.001816
grad_step = 000183, loss = 0.001804
grad_step = 000184, loss = 0.001805
grad_step = 000185, loss = 0.001813
grad_step = 000186, loss = 0.001820
grad_step = 000187, loss = 0.001824
grad_step = 000188, loss = 0.001818
grad_step = 000189, loss = 0.001809
grad_step = 000190, loss = 0.001798
grad_step = 000191, loss = 0.001791
grad_step = 000192, loss = 0.001790
grad_step = 000193, loss = 0.001793
grad_step = 000194, loss = 0.001800
grad_step = 000195, loss = 0.001808
grad_step = 000196, loss = 0.001817
grad_step = 000197, loss = 0.001823
grad_step = 000198, loss = 0.001826
grad_step = 000199, loss = 0.001823
grad_step = 000200, loss = 0.001815
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001802
grad_step = 000202, loss = 0.001786
grad_step = 000203, loss = 0.001773
grad_step = 000204, loss = 0.001765
grad_step = 000205, loss = 0.001762
grad_step = 000206, loss = 0.001764
grad_step = 000207, loss = 0.001768
grad_step = 000208, loss = 0.001774
grad_step = 000209, loss = 0.001780
grad_step = 000210, loss = 0.001790
grad_step = 000211, loss = 0.001810
grad_step = 000212, loss = 0.001852
grad_step = 000213, loss = 0.001949
grad_step = 000214, loss = 0.002071
grad_step = 000215, loss = 0.002179
grad_step = 000216, loss = 0.002041
grad_step = 000217, loss = 0.001824
grad_step = 000218, loss = 0.001786
grad_step = 000219, loss = 0.001910
grad_step = 000220, loss = 0.001934
grad_step = 000221, loss = 0.001796
grad_step = 000222, loss = 0.001761
grad_step = 000223, loss = 0.001865
grad_step = 000224, loss = 0.001881
grad_step = 000225, loss = 0.001788
grad_step = 000226, loss = 0.001762
grad_step = 000227, loss = 0.001826
grad_step = 000228, loss = 0.001828
grad_step = 000229, loss = 0.001755
grad_step = 000230, loss = 0.001735
grad_step = 000231, loss = 0.001782
grad_step = 000232, loss = 0.001790
grad_step = 000233, loss = 0.001751
grad_step = 000234, loss = 0.001731
grad_step = 000235, loss = 0.001758
grad_step = 000236, loss = 0.001781
grad_step = 000237, loss = 0.001762
grad_step = 000238, loss = 0.001740
grad_step = 000239, loss = 0.001752
grad_step = 000240, loss = 0.001781
grad_step = 000241, loss = 0.001789
grad_step = 000242, loss = 0.001781
grad_step = 000243, loss = 0.001782
grad_step = 000244, loss = 0.001797
grad_step = 000245, loss = 0.001805
grad_step = 000246, loss = 0.001782
grad_step = 000247, loss = 0.001750
grad_step = 000248, loss = 0.001730
grad_step = 000249, loss = 0.001724
grad_step = 000250, loss = 0.001718
grad_step = 000251, loss = 0.001710
grad_step = 000252, loss = 0.001705
grad_step = 000253, loss = 0.001710
grad_step = 000254, loss = 0.001722
grad_step = 000255, loss = 0.001733
grad_step = 000256, loss = 0.001739
grad_step = 000257, loss = 0.001741
grad_step = 000258, loss = 0.001741
grad_step = 000259, loss = 0.001741
grad_step = 000260, loss = 0.001741
grad_step = 000261, loss = 0.001735
grad_step = 000262, loss = 0.001725
grad_step = 000263, loss = 0.001711
grad_step = 000264, loss = 0.001698
grad_step = 000265, loss = 0.001689
grad_step = 000266, loss = 0.001684
grad_step = 000267, loss = 0.001682
grad_step = 000268, loss = 0.001680
grad_step = 000269, loss = 0.001678
grad_step = 000270, loss = 0.001676
grad_step = 000271, loss = 0.001674
grad_step = 000272, loss = 0.001674
grad_step = 000273, loss = 0.001676
grad_step = 000274, loss = 0.001681
grad_step = 000275, loss = 0.001693
grad_step = 000276, loss = 0.001713
grad_step = 000277, loss = 0.001752
grad_step = 000278, loss = 0.001820
grad_step = 000279, loss = 0.001925
grad_step = 000280, loss = 0.002053
grad_step = 000281, loss = 0.002152
grad_step = 000282, loss = 0.002094
grad_step = 000283, loss = 0.001912
grad_step = 000284, loss = 0.001716
grad_step = 000285, loss = 0.001691
grad_step = 000286, loss = 0.001797
grad_step = 000287, loss = 0.001863
grad_step = 000288, loss = 0.001814
grad_step = 000289, loss = 0.001712
grad_step = 000290, loss = 0.001688
grad_step = 000291, loss = 0.001744
grad_step = 000292, loss = 0.001770
grad_step = 000293, loss = 0.001718
grad_step = 000294, loss = 0.001657
grad_step = 000295, loss = 0.001666
grad_step = 000296, loss = 0.001718
grad_step = 000297, loss = 0.001726
grad_step = 000298, loss = 0.001679
grad_step = 000299, loss = 0.001638
grad_step = 000300, loss = 0.001648
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001675
grad_step = 000302, loss = 0.001671
grad_step = 000303, loss = 0.001646
grad_step = 000304, loss = 0.001635
grad_step = 000305, loss = 0.001657
grad_step = 000306, loss = 0.001678
grad_step = 000307, loss = 0.001671
grad_step = 000308, loss = 0.001637
grad_step = 000309, loss = 0.001617
grad_step = 000310, loss = 0.001624
grad_step = 000311, loss = 0.001639
grad_step = 000312, loss = 0.001641
grad_step = 000313, loss = 0.001626
grad_step = 000314, loss = 0.001609
grad_step = 000315, loss = 0.001608
grad_step = 000316, loss = 0.001617
grad_step = 000317, loss = 0.001625
grad_step = 000318, loss = 0.001624
grad_step = 000319, loss = 0.001624
grad_step = 000320, loss = 0.001639
grad_step = 000321, loss = 0.001683
grad_step = 000322, loss = 0.001785
grad_step = 000323, loss = 0.001932
grad_step = 000324, loss = 0.002148
grad_step = 000325, loss = 0.002134
grad_step = 000326, loss = 0.001993
grad_step = 000327, loss = 0.001707
grad_step = 000328, loss = 0.001612
grad_step = 000329, loss = 0.001761
grad_step = 000330, loss = 0.001871
grad_step = 000331, loss = 0.001766
grad_step = 000332, loss = 0.001604
grad_step = 000333, loss = 0.001631
grad_step = 000334, loss = 0.001731
grad_step = 000335, loss = 0.001713
grad_step = 000336, loss = 0.001648
grad_step = 000337, loss = 0.001627
grad_step = 000338, loss = 0.001638
grad_step = 000339, loss = 0.001634
grad_step = 000340, loss = 0.001633
grad_step = 000341, loss = 0.001626
grad_step = 000342, loss = 0.001586
grad_step = 000343, loss = 0.001593
grad_step = 000344, loss = 0.001627
grad_step = 000345, loss = 0.001600
grad_step = 000346, loss = 0.001557
grad_step = 000347, loss = 0.001572
grad_step = 000348, loss = 0.001597
grad_step = 000349, loss = 0.001578
grad_step = 000350, loss = 0.001553
grad_step = 000351, loss = 0.001562
grad_step = 000352, loss = 0.001569
grad_step = 000353, loss = 0.001557
grad_step = 000354, loss = 0.001554
grad_step = 000355, loss = 0.001553
grad_step = 000356, loss = 0.001543
grad_step = 000357, loss = 0.001542
grad_step = 000358, loss = 0.001547
grad_step = 000359, loss = 0.001540
grad_step = 000360, loss = 0.001528
grad_step = 000361, loss = 0.001527
grad_step = 000362, loss = 0.001533
grad_step = 000363, loss = 0.001530
grad_step = 000364, loss = 0.001521
grad_step = 000365, loss = 0.001517
grad_step = 000366, loss = 0.001517
grad_step = 000367, loss = 0.001514
grad_step = 000368, loss = 0.001509
grad_step = 000369, loss = 0.001508
grad_step = 000370, loss = 0.001509
grad_step = 000371, loss = 0.001507
grad_step = 000372, loss = 0.001502
grad_step = 000373, loss = 0.001498
grad_step = 000374, loss = 0.001499
grad_step = 000375, loss = 0.001506
grad_step = 000376, loss = 0.001538
grad_step = 000377, loss = 0.001652
grad_step = 000378, loss = 0.002011
grad_step = 000379, loss = 0.002220
grad_step = 000380, loss = 0.002582
grad_step = 000381, loss = 0.002358
grad_step = 000382, loss = 0.002091
grad_step = 000383, loss = 0.001785
grad_step = 000384, loss = 0.001759
grad_step = 000385, loss = 0.001886
grad_step = 000386, loss = 0.001832
grad_step = 000387, loss = 0.001813
grad_step = 000388, loss = 0.001689
grad_step = 000389, loss = 0.001615
grad_step = 000390, loss = 0.001740
grad_step = 000391, loss = 0.001692
grad_step = 000392, loss = 0.001554
grad_step = 000393, loss = 0.001580
grad_step = 000394, loss = 0.001623
grad_step = 000395, loss = 0.001571
grad_step = 000396, loss = 0.001544
grad_step = 000397, loss = 0.001523
grad_step = 000398, loss = 0.001562
grad_step = 000399, loss = 0.001547
grad_step = 000400, loss = 0.001493
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001508
grad_step = 000402, loss = 0.001522
grad_step = 000403, loss = 0.001489
grad_step = 000404, loss = 0.001480
grad_step = 000405, loss = 0.001482
grad_step = 000406, loss = 0.001468
grad_step = 000407, loss = 0.001474
grad_step = 000408, loss = 0.001459
grad_step = 000409, loss = 0.001448
grad_step = 000410, loss = 0.001450
grad_step = 000411, loss = 0.001444
grad_step = 000412, loss = 0.001437
grad_step = 000413, loss = 0.001434
grad_step = 000414, loss = 0.001427
grad_step = 000415, loss = 0.001413
grad_step = 000416, loss = 0.001420
grad_step = 000417, loss = 0.001428
grad_step = 000418, loss = 0.001411
grad_step = 000419, loss = 0.001398
grad_step = 000420, loss = 0.001398
grad_step = 000421, loss = 0.001399
grad_step = 000422, loss = 0.001394
grad_step = 000423, loss = 0.001395
grad_step = 000424, loss = 0.001405
grad_step = 000425, loss = 0.001409
grad_step = 000426, loss = 0.001421
grad_step = 000427, loss = 0.001446
grad_step = 000428, loss = 0.001500
grad_step = 000429, loss = 0.001564
grad_step = 000430, loss = 0.001667
grad_step = 000431, loss = 0.001691
grad_step = 000432, loss = 0.001656
grad_step = 000433, loss = 0.001485
grad_step = 000434, loss = 0.001364
grad_step = 000435, loss = 0.001381
grad_step = 000436, loss = 0.001465
grad_step = 000437, loss = 0.001484
grad_step = 000438, loss = 0.001401
grad_step = 000439, loss = 0.001347
grad_step = 000440, loss = 0.001376
grad_step = 000441, loss = 0.001415
grad_step = 000442, loss = 0.001398
grad_step = 000443, loss = 0.001349
grad_step = 000444, loss = 0.001335
grad_step = 000445, loss = 0.001355
grad_step = 000446, loss = 0.001368
grad_step = 000447, loss = 0.001359
grad_step = 000448, loss = 0.001331
grad_step = 000449, loss = 0.001312
grad_step = 000450, loss = 0.001317
grad_step = 000451, loss = 0.001332
grad_step = 000452, loss = 0.001336
grad_step = 000453, loss = 0.001319
grad_step = 000454, loss = 0.001297
grad_step = 000455, loss = 0.001286
grad_step = 000456, loss = 0.001287
grad_step = 000457, loss = 0.001293
grad_step = 000458, loss = 0.001297
grad_step = 000459, loss = 0.001302
grad_step = 000460, loss = 0.001311
grad_step = 000461, loss = 0.001329
grad_step = 000462, loss = 0.001337
grad_step = 000463, loss = 0.001359
grad_step = 000464, loss = 0.001333
grad_step = 000465, loss = 0.001328
grad_step = 000466, loss = 0.001287
grad_step = 000467, loss = 0.001256
grad_step = 000468, loss = 0.001241
grad_step = 000469, loss = 0.001246
grad_step = 000470, loss = 0.001260
grad_step = 000471, loss = 0.001261
grad_step = 000472, loss = 0.001258
grad_step = 000473, loss = 0.001241
grad_step = 000474, loss = 0.001236
grad_step = 000475, loss = 0.001230
grad_step = 000476, loss = 0.001247
grad_step = 000477, loss = 0.001270
grad_step = 000478, loss = 0.001364
grad_step = 000479, loss = 0.001424
grad_step = 000480, loss = 0.001621
grad_step = 000481, loss = 0.001503
grad_step = 000482, loss = 0.001403
grad_step = 000483, loss = 0.001257
grad_step = 000484, loss = 0.001230
grad_step = 000485, loss = 0.001306
grad_step = 000486, loss = 0.001347
grad_step = 000487, loss = 0.001286
grad_step = 000488, loss = 0.001195
grad_step = 000489, loss = 0.001187
grad_step = 000490, loss = 0.001206
grad_step = 000491, loss = 0.001205
grad_step = 000492, loss = 0.001256
grad_step = 000493, loss = 0.001428
grad_step = 000494, loss = 0.001756
grad_step = 000495, loss = 0.001882
grad_step = 000496, loss = 0.001932
grad_step = 000497, loss = 0.001618
grad_step = 000498, loss = 0.001244
grad_step = 000499, loss = 0.001223
grad_step = 000500, loss = 0.001483
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001487
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

  date_run                              2020-05-09 14:43:31.868618
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.283609
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 14:43:31.874159
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.210127
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 14:43:31.880841
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.15786
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 14:43:31.885162
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -2.19295
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
0   2020-05-09 14:43:08.690017  ...    mean_absolute_error
1   2020-05-09 14:43:08.693219  ...     mean_squared_error
2   2020-05-09 14:43:08.696053  ...  median_absolute_error
3   2020-05-09 14:43:08.699053  ...               r2_score
4   2020-05-09 14:43:15.879990  ...    mean_absolute_error
5   2020-05-09 14:43:15.883615  ...     mean_squared_error
6   2020-05-09 14:43:15.886896  ...  median_absolute_error
7   2020-05-09 14:43:15.889627  ...               r2_score
8   2020-05-09 14:43:31.868618  ...    mean_absolute_error
9   2020-05-09 14:43:31.874159  ...     mean_squared_error
10  2020-05-09 14:43:31.880841  ...  median_absolute_error
11  2020-05-09 14:43:31.885162  ...               r2_score

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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140349145817888
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140347688726312
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140347688276320
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140347688276824
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140347688277328
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140347688277832

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa594fb2400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.567938
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.539192
grad_step = 000002, loss = 0.520478
grad_step = 000003, loss = 0.501472
grad_step = 000004, loss = 0.482248
grad_step = 000005, loss = 0.463897
grad_step = 000006, loss = 0.443529
grad_step = 000007, loss = 0.425851
grad_step = 000008, loss = 0.407991
grad_step = 000009, loss = 0.389091
grad_step = 000010, loss = 0.371578
grad_step = 000011, loss = 0.357639
grad_step = 000012, loss = 0.346142
grad_step = 000013, loss = 0.335098
grad_step = 000014, loss = 0.323214
grad_step = 000015, loss = 0.309942
grad_step = 000016, loss = 0.296816
grad_step = 000017, loss = 0.284281
grad_step = 000018, loss = 0.271985
grad_step = 000019, loss = 0.259589
grad_step = 000020, loss = 0.247513
grad_step = 000021, loss = 0.235840
grad_step = 000022, loss = 0.224288
grad_step = 000023, loss = 0.212842
grad_step = 000024, loss = 0.201744
grad_step = 000025, loss = 0.190937
grad_step = 000026, loss = 0.180818
grad_step = 000027, loss = 0.170187
grad_step = 000028, loss = 0.159443
grad_step = 000029, loss = 0.149209
grad_step = 000030, loss = 0.139561
grad_step = 000031, loss = 0.130539
grad_step = 000032, loss = 0.122377
grad_step = 000033, loss = 0.114735
grad_step = 000034, loss = 0.107412
grad_step = 000035, loss = 0.100319
grad_step = 000036, loss = 0.093576
grad_step = 000037, loss = 0.087332
grad_step = 000038, loss = 0.081466
grad_step = 000039, loss = 0.075723
grad_step = 000040, loss = 0.070278
grad_step = 000041, loss = 0.065217
grad_step = 000042, loss = 0.060489
grad_step = 000043, loss = 0.056097
grad_step = 000044, loss = 0.051930
grad_step = 000045, loss = 0.048087
grad_step = 000046, loss = 0.044543
grad_step = 000047, loss = 0.041300
grad_step = 000048, loss = 0.038247
grad_step = 000049, loss = 0.035357
grad_step = 000050, loss = 0.032645
grad_step = 000051, loss = 0.030089
grad_step = 000052, loss = 0.027752
grad_step = 000053, loss = 0.025607
grad_step = 000054, loss = 0.023610
grad_step = 000055, loss = 0.021735
grad_step = 000056, loss = 0.019990
grad_step = 000057, loss = 0.018351
grad_step = 000058, loss = 0.016828
grad_step = 000059, loss = 0.015433
grad_step = 000060, loss = 0.014144
grad_step = 000061, loss = 0.012948
grad_step = 000062, loss = 0.011835
grad_step = 000063, loss = 0.010815
grad_step = 000064, loss = 0.009889
grad_step = 000065, loss = 0.009042
grad_step = 000066, loss = 0.008263
grad_step = 000067, loss = 0.007557
grad_step = 000068, loss = 0.006913
grad_step = 000069, loss = 0.006328
grad_step = 000070, loss = 0.005798
grad_step = 000071, loss = 0.005323
grad_step = 000072, loss = 0.004903
grad_step = 000073, loss = 0.004534
grad_step = 000074, loss = 0.004207
grad_step = 000075, loss = 0.003915
grad_step = 000076, loss = 0.003663
grad_step = 000077, loss = 0.003444
grad_step = 000078, loss = 0.003249
grad_step = 000079, loss = 0.003080
grad_step = 000080, loss = 0.002937
grad_step = 000081, loss = 0.002817
grad_step = 000082, loss = 0.002715
grad_step = 000083, loss = 0.002630
grad_step = 000084, loss = 0.002560
grad_step = 000085, loss = 0.002502
grad_step = 000086, loss = 0.002453
grad_step = 000087, loss = 0.002411
grad_step = 000088, loss = 0.002377
grad_step = 000089, loss = 0.002349
grad_step = 000090, loss = 0.002325
grad_step = 000091, loss = 0.002305
grad_step = 000092, loss = 0.002290
grad_step = 000093, loss = 0.002276
grad_step = 000094, loss = 0.002265
grad_step = 000095, loss = 0.002254
grad_step = 000096, loss = 0.002244
grad_step = 000097, loss = 0.002235
grad_step = 000098, loss = 0.002227
grad_step = 000099, loss = 0.002221
grad_step = 000100, loss = 0.002219
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002226
grad_step = 000102, loss = 0.002252
grad_step = 000103, loss = 0.002298
grad_step = 000104, loss = 0.002351
grad_step = 000105, loss = 0.002329
grad_step = 000106, loss = 0.002243
grad_step = 000107, loss = 0.002163
grad_step = 000108, loss = 0.002171
grad_step = 000109, loss = 0.002227
grad_step = 000110, loss = 0.002235
grad_step = 000111, loss = 0.002182
grad_step = 000112, loss = 0.002128
grad_step = 000113, loss = 0.002134
grad_step = 000114, loss = 0.002170
grad_step = 000115, loss = 0.002171
grad_step = 000116, loss = 0.002134
grad_step = 000117, loss = 0.002099
grad_step = 000118, loss = 0.002100
grad_step = 000119, loss = 0.002121
grad_step = 000120, loss = 0.002125
grad_step = 000121, loss = 0.002106
grad_step = 000122, loss = 0.002078
grad_step = 000123, loss = 0.002068
grad_step = 000124, loss = 0.002075
grad_step = 000125, loss = 0.002085
grad_step = 000126, loss = 0.002086
grad_step = 000127, loss = 0.002075
grad_step = 000128, loss = 0.002058
grad_step = 000129, loss = 0.002044
grad_step = 000130, loss = 0.002037
grad_step = 000131, loss = 0.002036
grad_step = 000132, loss = 0.002039
grad_step = 000133, loss = 0.002045
grad_step = 000134, loss = 0.002052
grad_step = 000135, loss = 0.002062
grad_step = 000136, loss = 0.002073
grad_step = 000137, loss = 0.002091
grad_step = 000138, loss = 0.002105
grad_step = 000139, loss = 0.002120
grad_step = 000140, loss = 0.002107
grad_step = 000141, loss = 0.002082
grad_step = 000142, loss = 0.002038
grad_step = 000143, loss = 0.002003
grad_step = 000144, loss = 0.001985
grad_step = 000145, loss = 0.001987
grad_step = 000146, loss = 0.002003
grad_step = 000147, loss = 0.002026
grad_step = 000148, loss = 0.002054
grad_step = 000149, loss = 0.002076
grad_step = 000150, loss = 0.002097
grad_step = 000151, loss = 0.002089
grad_step = 000152, loss = 0.002066
grad_step = 000153, loss = 0.002018
grad_step = 000154, loss = 0.001976
grad_step = 000155, loss = 0.001952
grad_step = 000156, loss = 0.001950
grad_step = 000157, loss = 0.001965
grad_step = 000158, loss = 0.001987
grad_step = 000159, loss = 0.002014
grad_step = 000160, loss = 0.002031
grad_step = 000161, loss = 0.002044
grad_step = 000162, loss = 0.002032
grad_step = 000163, loss = 0.002011
grad_step = 000164, loss = 0.001974
grad_step = 000165, loss = 0.001942
grad_step = 000166, loss = 0.001920
grad_step = 000167, loss = 0.001913
grad_step = 000168, loss = 0.001918
grad_step = 000169, loss = 0.001931
grad_step = 000170, loss = 0.001953
grad_step = 000171, loss = 0.001983
grad_step = 000172, loss = 0.002029
grad_step = 000173, loss = 0.002076
grad_step = 000174, loss = 0.002139
grad_step = 000175, loss = 0.002136
grad_step = 000176, loss = 0.002096
grad_step = 000177, loss = 0.001983
grad_step = 000178, loss = 0.001899
grad_step = 000179, loss = 0.001885
grad_step = 000180, loss = 0.001933
grad_step = 000181, loss = 0.001993
grad_step = 000182, loss = 0.002001
grad_step = 000183, loss = 0.001967
grad_step = 000184, loss = 0.001904
grad_step = 000185, loss = 0.001866
grad_step = 000186, loss = 0.001868
grad_step = 000187, loss = 0.001898
grad_step = 000188, loss = 0.001930
grad_step = 000189, loss = 0.001940
grad_step = 000190, loss = 0.001929
grad_step = 000191, loss = 0.001896
grad_step = 000192, loss = 0.001864
grad_step = 000193, loss = 0.001844
grad_step = 000194, loss = 0.001840
grad_step = 000195, loss = 0.001849
grad_step = 000196, loss = 0.001864
grad_step = 000197, loss = 0.001881
grad_step = 000198, loss = 0.001892
grad_step = 000199, loss = 0.001905
grad_step = 000200, loss = 0.001908
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001912
grad_step = 000202, loss = 0.001897
grad_step = 000203, loss = 0.001888
grad_step = 000204, loss = 0.001873
grad_step = 000205, loss = 0.001862
grad_step = 000206, loss = 0.001847
grad_step = 000207, loss = 0.001836
grad_step = 000208, loss = 0.001824
grad_step = 000209, loss = 0.001817
grad_step = 000210, loss = 0.001811
grad_step = 000211, loss = 0.001808
grad_step = 000212, loss = 0.001807
grad_step = 000213, loss = 0.001812
grad_step = 000214, loss = 0.001822
grad_step = 000215, loss = 0.001849
grad_step = 000216, loss = 0.001888
grad_step = 000217, loss = 0.001966
grad_step = 000218, loss = 0.002039
grad_step = 000219, loss = 0.002134
grad_step = 000220, loss = 0.002072
grad_step = 000221, loss = 0.001932
grad_step = 000222, loss = 0.001785
grad_step = 000223, loss = 0.001768
grad_step = 000224, loss = 0.001856
grad_step = 000225, loss = 0.001923
grad_step = 000226, loss = 0.001904
grad_step = 000227, loss = 0.001803
grad_step = 000228, loss = 0.001738
grad_step = 000229, loss = 0.001751
grad_step = 000230, loss = 0.001806
grad_step = 000231, loss = 0.001852
grad_step = 000232, loss = 0.001834
grad_step = 000233, loss = 0.001786
grad_step = 000234, loss = 0.001730
grad_step = 000235, loss = 0.001706
grad_step = 000236, loss = 0.001721
grad_step = 000237, loss = 0.001751
grad_step = 000238, loss = 0.001773
grad_step = 000239, loss = 0.001764
grad_step = 000240, loss = 0.001740
grad_step = 000241, loss = 0.001706
grad_step = 000242, loss = 0.001681
grad_step = 000243, loss = 0.001671
grad_step = 000244, loss = 0.001675
grad_step = 000245, loss = 0.001686
grad_step = 000246, loss = 0.001700
grad_step = 000247, loss = 0.001722
grad_step = 000248, loss = 0.001746
grad_step = 000249, loss = 0.001772
grad_step = 000250, loss = 0.001767
grad_step = 000251, loss = 0.001756
grad_step = 000252, loss = 0.001721
grad_step = 000253, loss = 0.001678
grad_step = 000254, loss = 0.001636
grad_step = 000255, loss = 0.001623
grad_step = 000256, loss = 0.001631
grad_step = 000257, loss = 0.001642
grad_step = 000258, loss = 0.001663
grad_step = 000259, loss = 0.001694
grad_step = 000260, loss = 0.001747
grad_step = 000261, loss = 0.001784
grad_step = 000262, loss = 0.001846
grad_step = 000263, loss = 0.001837
grad_step = 000264, loss = 0.001796
grad_step = 000265, loss = 0.001680
grad_step = 000266, loss = 0.001600
grad_step = 000267, loss = 0.001592
grad_step = 000268, loss = 0.001636
grad_step = 000269, loss = 0.001688
grad_step = 000270, loss = 0.001689
grad_step = 000271, loss = 0.001648
grad_step = 000272, loss = 0.001592
grad_step = 000273, loss = 0.001564
grad_step = 000274, loss = 0.001566
grad_step = 000275, loss = 0.001593
grad_step = 000276, loss = 0.001634
grad_step = 000277, loss = 0.001660
grad_step = 000278, loss = 0.001689
grad_step = 000279, loss = 0.001682
grad_step = 000280, loss = 0.001656
grad_step = 000281, loss = 0.001603
grad_step = 000282, loss = 0.001558
grad_step = 000283, loss = 0.001537
grad_step = 000284, loss = 0.001548
grad_step = 000285, loss = 0.001574
grad_step = 000286, loss = 0.001588
grad_step = 000287, loss = 0.001584
grad_step = 000288, loss = 0.001564
grad_step = 000289, loss = 0.001541
grad_step = 000290, loss = 0.001522
grad_step = 000291, loss = 0.001515
grad_step = 000292, loss = 0.001519
grad_step = 000293, loss = 0.001527
grad_step = 000294, loss = 0.001542
grad_step = 000295, loss = 0.001559
grad_step = 000296, loss = 0.001586
grad_step = 000297, loss = 0.001612
grad_step = 000298, loss = 0.001651
grad_step = 000299, loss = 0.001667
grad_step = 000300, loss = 0.001672
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001628
grad_step = 000302, loss = 0.001568
grad_step = 000303, loss = 0.001510
grad_step = 000304, loss = 0.001487
grad_step = 000305, loss = 0.001500
grad_step = 000306, loss = 0.001531
grad_step = 000307, loss = 0.001563
grad_step = 000308, loss = 0.001573
grad_step = 000309, loss = 0.001567
grad_step = 000310, loss = 0.001539
grad_step = 000311, loss = 0.001507
grad_step = 000312, loss = 0.001480
grad_step = 000313, loss = 0.001465
grad_step = 000314, loss = 0.001464
grad_step = 000315, loss = 0.001472
grad_step = 000316, loss = 0.001486
grad_step = 000317, loss = 0.001502
grad_step = 000318, loss = 0.001522
grad_step = 000319, loss = 0.001539
grad_step = 000320, loss = 0.001562
grad_step = 000321, loss = 0.001573
grad_step = 000322, loss = 0.001583
grad_step = 000323, loss = 0.001564
grad_step = 000324, loss = 0.001533
grad_step = 000325, loss = 0.001487
grad_step = 000326, loss = 0.001450
grad_step = 000327, loss = 0.001432
grad_step = 000328, loss = 0.001436
grad_step = 000329, loss = 0.001453
grad_step = 000330, loss = 0.001472
grad_step = 000331, loss = 0.001489
grad_step = 000332, loss = 0.001498
grad_step = 000333, loss = 0.001503
grad_step = 000334, loss = 0.001495
grad_step = 000335, loss = 0.001484
grad_step = 000336, loss = 0.001463
grad_step = 000337, loss = 0.001442
grad_step = 000338, loss = 0.001422
grad_step = 000339, loss = 0.001407
grad_step = 000340, loss = 0.001399
grad_step = 000341, loss = 0.001397
grad_step = 000342, loss = 0.001399
grad_step = 000343, loss = 0.001404
grad_step = 000344, loss = 0.001414
grad_step = 000345, loss = 0.001430
grad_step = 000346, loss = 0.001459
grad_step = 000347, loss = 0.001501
grad_step = 000348, loss = 0.001576
grad_step = 000349, loss = 0.001651
grad_step = 000350, loss = 0.001726
grad_step = 000351, loss = 0.001704
grad_step = 000352, loss = 0.001598
grad_step = 000353, loss = 0.001448
grad_step = 000354, loss = 0.001371
grad_step = 000355, loss = 0.001408
grad_step = 000356, loss = 0.001491
grad_step = 000357, loss = 0.001526
grad_step = 000358, loss = 0.001475
grad_step = 000359, loss = 0.001397
grad_step = 000360, loss = 0.001356
grad_step = 000361, loss = 0.001372
grad_step = 000362, loss = 0.001414
grad_step = 000363, loss = 0.001441
grad_step = 000364, loss = 0.001439
grad_step = 000365, loss = 0.001403
grad_step = 000366, loss = 0.001368
grad_step = 000367, loss = 0.001347
grad_step = 000368, loss = 0.001349
grad_step = 000369, loss = 0.001364
grad_step = 000370, loss = 0.001377
grad_step = 000371, loss = 0.001379
grad_step = 000372, loss = 0.001366
grad_step = 000373, loss = 0.001350
grad_step = 000374, loss = 0.001335
grad_step = 000375, loss = 0.001328
grad_step = 000376, loss = 0.001326
grad_step = 000377, loss = 0.001329
grad_step = 000378, loss = 0.001333
grad_step = 000379, loss = 0.001336
grad_step = 000380, loss = 0.001339
grad_step = 000381, loss = 0.001339
grad_step = 000382, loss = 0.001339
grad_step = 000383, loss = 0.001336
grad_step = 000384, loss = 0.001334
grad_step = 000385, loss = 0.001330
grad_step = 000386, loss = 0.001327
grad_step = 000387, loss = 0.001325
grad_step = 000388, loss = 0.001327
grad_step = 000389, loss = 0.001327
grad_step = 000390, loss = 0.001329
grad_step = 000391, loss = 0.001327
grad_step = 000392, loss = 0.001327
grad_step = 000393, loss = 0.001328
grad_step = 000394, loss = 0.001330
grad_step = 000395, loss = 0.001337
grad_step = 000396, loss = 0.001341
grad_step = 000397, loss = 0.001348
grad_step = 000398, loss = 0.001344
grad_step = 000399, loss = 0.001338
grad_step = 000400, loss = 0.001318
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001300
grad_step = 000402, loss = 0.001283
grad_step = 000403, loss = 0.001277
grad_step = 000404, loss = 0.001282
grad_step = 000405, loss = 0.001297
grad_step = 000406, loss = 0.001320
grad_step = 000407, loss = 0.001350
grad_step = 000408, loss = 0.001409
grad_step = 000409, loss = 0.001529
grad_step = 000410, loss = 0.001644
grad_step = 000411, loss = 0.001657
grad_step = 000412, loss = 0.001490
grad_step = 000413, loss = 0.001367
grad_step = 000414, loss = 0.001406
grad_step = 000415, loss = 0.001404
grad_step = 000416, loss = 0.001298
grad_step = 000417, loss = 0.001312
grad_step = 000418, loss = 0.001403
grad_step = 000419, loss = 0.001370
grad_step = 000420, loss = 0.001294
grad_step = 000421, loss = 0.001303
grad_step = 000422, loss = 0.001306
grad_step = 000423, loss = 0.001275
grad_step = 000424, loss = 0.001283
grad_step = 000425, loss = 0.001318
grad_step = 000426, loss = 0.001296
grad_step = 000427, loss = 0.001256
grad_step = 000428, loss = 0.001260
grad_step = 000429, loss = 0.001274
grad_step = 000430, loss = 0.001254
grad_step = 000431, loss = 0.001247
grad_step = 000432, loss = 0.001266
grad_step = 000433, loss = 0.001273
grad_step = 000434, loss = 0.001256
grad_step = 000435, loss = 0.001248
grad_step = 000436, loss = 0.001251
grad_step = 000437, loss = 0.001247
grad_step = 000438, loss = 0.001234
grad_step = 000439, loss = 0.001228
grad_step = 000440, loss = 0.001231
grad_step = 000441, loss = 0.001233
grad_step = 000442, loss = 0.001227
grad_step = 000443, loss = 0.001223
grad_step = 000444, loss = 0.001228
grad_step = 000445, loss = 0.001231
grad_step = 000446, loss = 0.001228
grad_step = 000447, loss = 0.001227
grad_step = 000448, loss = 0.001231
grad_step = 000449, loss = 0.001236
grad_step = 000450, loss = 0.001240
grad_step = 000451, loss = 0.001244
grad_step = 000452, loss = 0.001254
grad_step = 000453, loss = 0.001265
grad_step = 000454, loss = 0.001280
grad_step = 000455, loss = 0.001284
grad_step = 000456, loss = 0.001292
grad_step = 000457, loss = 0.001289
grad_step = 000458, loss = 0.001283
grad_step = 000459, loss = 0.001263
grad_step = 000460, loss = 0.001239
grad_step = 000461, loss = 0.001214
grad_step = 000462, loss = 0.001200
grad_step = 000463, loss = 0.001196
grad_step = 000464, loss = 0.001202
grad_step = 000465, loss = 0.001213
grad_step = 000466, loss = 0.001222
grad_step = 000467, loss = 0.001230
grad_step = 000468, loss = 0.001232
grad_step = 000469, loss = 0.001233
grad_step = 000470, loss = 0.001229
grad_step = 000471, loss = 0.001224
grad_step = 000472, loss = 0.001214
grad_step = 000473, loss = 0.001206
grad_step = 000474, loss = 0.001197
grad_step = 000475, loss = 0.001189
grad_step = 000476, loss = 0.001184
grad_step = 000477, loss = 0.001179
grad_step = 000478, loss = 0.001176
grad_step = 000479, loss = 0.001175
grad_step = 000480, loss = 0.001175
grad_step = 000481, loss = 0.001176
grad_step = 000482, loss = 0.001178
grad_step = 000483, loss = 0.001181
grad_step = 000484, loss = 0.001187
grad_step = 000485, loss = 0.001196
grad_step = 000486, loss = 0.001213
grad_step = 000487, loss = 0.001235
grad_step = 000488, loss = 0.001274
grad_step = 000489, loss = 0.001312
grad_step = 000490, loss = 0.001371
grad_step = 000491, loss = 0.001394
grad_step = 000492, loss = 0.001416
grad_step = 000493, loss = 0.001369
grad_step = 000494, loss = 0.001307
grad_step = 000495, loss = 0.001249
grad_step = 000496, loss = 0.001211
grad_step = 000497, loss = 0.001200
grad_step = 000498, loss = 0.001197
grad_step = 000499, loss = 0.001203
grad_step = 000500, loss = 0.001212
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001217
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

  date_run                              2020-05-09 14:43:50.477111
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.278385
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-09 14:43:50.482412
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.20394
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-09 14:43:50.489162
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.14634
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-09 14:43:50.494142
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -2.09895
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa594fb2390> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 362527.5938
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 327969.0625
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 269303.9062
Epoch 4/10

1/1 [==============================] - 0s 83ms/step - loss: 188737.6562
Epoch 5/10

1/1 [==============================] - 0s 82ms/step - loss: 121066.4141
Epoch 6/10

1/1 [==============================] - 0s 80ms/step - loss: 70846.3516
Epoch 7/10

1/1 [==============================] - 0s 82ms/step - loss: 41252.4102
Epoch 8/10

1/1 [==============================] - 0s 89ms/step - loss: 24959.9004
Epoch 9/10

1/1 [==============================] - 0s 75ms/step - loss: 16021.6387
Epoch 10/10

1/1 [==============================] - 0s 78ms/step - loss: 10945.6016

  #### Inference Need return ypred, ytrue ######################### 
[[-0.02252816  6.5308304   6.0736213   6.55212     5.564288    5.9482856
   6.25059     5.483342    7.53684     6.125731    5.6881423   5.6859965
   6.546936    5.702377    5.5747848   6.450814    6.4059763   5.440823
   7.5658636   5.5769353   6.642077    5.850892    5.905826    5.099402
   7.0249143   4.721327    5.739714    5.8737297   4.028724    6.1258917
   6.541816    6.3291736   5.554042    7.416083    6.3725696   4.6312675
   6.2253194   6.8493357   6.054471    5.4451466   6.046418    6.3197565
   5.0974727   4.955988    5.93068     5.8572464   5.8730097   7.085741
   5.494689    4.373559    6.9816947   7.1404314   5.7771764   5.695047
   6.6077933   6.691316    5.612506    5.7452435   5.800501    6.1534767
  -0.38842452 -0.6034955  -0.6012015  -1.3798397   0.4651418  -0.82757133
   1.5730996  -0.03096592 -0.9685525  -0.10732742  0.33434874 -0.38694
  -1.4567285  -0.33540568  0.8807087  -0.46287206  0.63866043  0.48503184
  -0.8029387   0.14994103  0.513299    0.4786287   1.4384018   1.126359
   0.4611046   0.03944847  1.0030384   0.45123014 -0.3006118   0.7142751
   0.5091435   0.26068717  1.8756287  -0.68893856  0.6478689   0.26507616
   1.1546118  -1.2753778   0.5172714  -0.4197692  -0.27865607  1.1073511
  -0.57735616  0.14061433 -1.085767   -0.6176868   0.32541975  0.4356274
   0.1024701   0.33333895  0.21740067 -0.8751601  -0.34152436  0.06440881
  -1.791609   -0.35161358  0.66540074  1.7322379  -0.7945938   0.11788559
  -0.61217076 -0.4173718   0.8976319   0.8002387  -1.8510232   1.511043
   1.4964042  -0.17633832 -0.47633088 -0.44181907  0.61920774  1.4609616
   0.03135848 -0.4823049  -0.5705074  -0.9774216  -0.13760321  0.9413577
   0.2452685  -0.50319046  0.54233277  0.8766983   0.79788214  0.45848486
   0.08722824 -0.3467216  -0.6536747  -0.5708327  -0.9915662   0.45836982
   0.7964102  -0.25015888  0.447221    0.5676085   0.06964278 -0.2384974
   1.1111494  -0.8786414   0.07336536 -0.17572147 -0.5193366   0.19424129
  -1.6294384  -0.73615307 -0.42992806 -1.0075523  -0.9847933   1.3868889
  -1.8043145  -0.62718374  1.0895354   0.9750814  -0.13913888  0.04250091
  -0.13246444  0.15550876  1.826579    0.16488343  0.11467531 -0.25528413
   0.03699452  6.986371    6.2103024   6.743699    5.906233    6.722127
   7.003772    6.924749    6.7176013   7.3791966   7.381636    6.364651
   5.738395    7.193251    6.345575    7.028031    6.6808524   5.3918304
   6.544239    6.913562    7.251623    5.8804398   6.068243    6.832154
   7.438306    6.168311    6.878082    6.5024056   6.9174404   7.1485634
   6.654703    7.206071    6.079437    7.320444    6.5372615   7.275118
   6.733983    7.3100333   6.782256    7.6614614   6.1588306   7.729695
   8.252667    7.415626    7.47861     6.5849514   6.9760566   6.1510515
   6.869378    7.5069046   5.7338963   7.2601004   5.9616814   7.1779504
   7.303769    7.0594144   6.3659334   7.029693    6.4676814   7.2833605
   0.7520747   1.7225993   2.0169413   0.38504165  1.3575776   0.70269966
   1.242276    0.7371986   2.0177364   1.6445076   0.7429484   1.2259626
   0.87447596  1.05232     1.6396027   1.0751295   1.3108948   1.9881067
   0.3182826   0.43994218  1.8081902   2.1210148   1.8346043   1.5549495
   1.7058439   1.0604333   1.9690828   0.83562493  1.0545971   0.3010695
   1.125709    1.1512041   1.560505    0.45255727  0.9061382   0.89967746
   0.7586709   1.5411446   1.5111254   0.39279443  1.8171296   0.6079346
   0.6813598   1.4462361   0.61366117  1.7111969   0.6162956   0.22858536
   0.9877281   1.2706398   1.4313139   0.44584638  1.3695309   1.8747438
   1.8729342   1.2653704   0.46597242  0.8671284   1.1315022   1.7226958
   1.5158997   1.0824292   0.3090791   0.561268    1.3146663   1.6138382
   0.8543839   0.4700725   1.5531902   0.46739316  1.6967415   0.8103572
   0.3611216   0.9572457   1.0636365   1.0952727   1.1254035   1.8133624
   1.3179399   0.28631604  1.0598764   1.2890253   1.6677722   2.8898034
   0.5358783   0.78429115  2.0464854   1.7077245   1.234329    0.3235879
   0.8042255   0.46205437  1.0745566   0.39743817  0.95452845  0.51390725
   0.45717466  0.44575113  0.7752444   1.1182883   1.4281744   0.39459032
   1.2671969   1.3316977   0.81826186  1.2740836   1.3605301   0.4262241
   1.0211575   0.5375337   0.5908757   1.8304658   0.49970585  1.9971701
   0.6112678   0.57807124  1.0516357   1.9872514   0.57757676  1.2791831
   2.828587   -5.2522492  -5.3097873 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:43:57.719216
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    96.125
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-09 14:43:57.722499
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9259.11
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-09 14:43:57.724984
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    95.929
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-09 14:43:57.727159
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -828.219
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa525d13e10> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-09 14:44:10.560550
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-09 14:44:10.563600
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-09 14:44:10.566111
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-09 14:44:10.568785
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
0   2020-05-09 14:43:50.477111  ...    mean_absolute_error
1   2020-05-09 14:43:50.482412  ...     mean_squared_error
2   2020-05-09 14:43:50.489162  ...  median_absolute_error
3   2020-05-09 14:43:50.494142  ...               r2_score
4   2020-05-09 14:43:57.719216  ...    mean_absolute_error
5   2020-05-09 14:43:57.722499  ...     mean_squared_error
6   2020-05-09 14:43:57.724984  ...  median_absolute_error
7   2020-05-09 14:43:57.727159  ...               r2_score
8   2020-05-09 14:44:10.560550  ...    mean_absolute_error
9   2020-05-09 14:44:10.563600  ...     mean_squared_error
10  2020-05-09 14:44:10.566111  ...  median_absolute_error
11  2020-05-09 14:44:10.568785  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 118, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

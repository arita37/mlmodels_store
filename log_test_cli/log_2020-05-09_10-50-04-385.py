  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 
  test_cli GITHUB_REPOSITORT GITHUB_SHA 
  Running command test_cli 
  # Testing Command Line System   





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/378ea73103703a153c9dba7ff24d59bc1ff754dd', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '378ea73103703a153c9dba7ff24d59bc1ff754dd', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/378ea73103703a153c9dba7ff24d59bc1ff754dd

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/378ea73103703a153c9dba7ff24d59bc1ff754dd

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
  <mlmodels.model_tf.1_lstm.Model object at 0x7f5adca387b8> 
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
 [-0.20993704  0.09496422  0.18687844  0.17330183 -0.05268951 -0.02770817]
 [-0.15820169  0.07820895  0.02199633 -0.02382268  0.07520352  0.06520255]
 [-0.00401762  0.04452737  0.0521032   0.07835872  0.08718531  0.0864317 ]
 [ 0.40158403 -0.27390498  0.0900498  -0.11019678  0.27966031  0.04444347]
 [ 0.32874179 -0.455293   -0.39595404 -0.10014007  0.40449598  0.34185401]
 [ 0.1678483   0.07854903  0.01249661 -0.10610433  0.10661573  0.36461791]
 [-0.42397588  0.24934062  0.23969793 -0.03249391 -0.06073882  0.33968407]
 [-0.55202937  0.82340908 -0.14358692 -0.24852736  0.70315492  0.44463965]
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
{'loss': 0.41529496759176254, 'loss_history': []}
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
{'loss': 0.4785858578979969, 'loss_history': []}
  #### Plot   ######################################################## 
  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





 ************************************************************************************************************************
ml_models --do test  --model_uri "ztest/mycustom/my_lstm.py"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
test
  #### Module init   ############################################ 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
    module = import_module(f"mlmodels.{model_name}")
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 941, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 950, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlmodels.ztest.mycustom'; 'mlmodels.ztest' is not a package

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 499, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 429, in test_cli
    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 250, in test_module
    module = module_load(model_uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module ztest.mycustom.my_lstm notfound, No module named 'mlmodels.ztest.mycustom'; 'mlmodels.ztest' is not a package, tuple index out of range





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
[32m[I 2020-05-09 10:50:32,752][0m Finished trial#0 resulted in value: 9.53827652335167. Current best value is 9.53827652335167 with parameters: {'learning_rate': 0.07041503813347975, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-09 10:50:34,718][0m Finished trial#1 resulted in value: 7.404086709022522. Current best value is 7.404086709022522 with parameters: {'learning_rate': 0.06396452550047114, 'num_layers': 4, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
  


### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_gluon/fb_prophet.py'} 
  #### Setup Model   ############################################## 
  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4ac8bf5be0> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} [Errno 2] File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist: b'dataset/timeseries/stock/qqq_us_train.csv' 
  


### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10} 
  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 124, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 89, in fit
    train_df, test_df = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 32, in get_dataset
    train_df = pd.read_csv(data_pars["train_data_path"], parse_dates=True)[col]
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
FileNotFoundError: [Errno 2] File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist: b'dataset/timeseries/stock/qqq_us_train.csv'
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4ab7f5b0f0> <class 'mlmodels.model_keras.armdn.Model'>
  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 356215.7500
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 309664.7188
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 225318.0312
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 151776.0312
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 96456.8203
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 60109.4141
Epoch 7/10

1/1 [==============================] - 0s 107ms/step - loss: 38314.2461
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 25582.0293
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 18015.4629
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 13344.8223
  #### Inference Need return ypred, ytrue ######################### 
[[ 1.3412982e-03  5.4027290e+00  4.3949518e+00  5.1217012e+00
   5.2159610e+00  6.0483360e+00  4.7151680e+00  4.8970599e+00
   4.0203180e+00  4.7304897e+00  5.4083366e+00  6.3741431e+00
   5.3276577e+00  4.9356441e+00  5.5163889e+00  4.5799298e+00
   5.0336432e+00  6.3252268e+00  6.0165215e+00  4.8086576e+00
   5.5811982e+00  5.0821829e+00  5.1845355e+00  4.4349980e+00
   5.0221310e+00  4.0795507e+00  4.3353515e+00  5.5397363e+00
   4.3666301e+00  6.0358353e+00  5.7228675e+00  5.9843049e+00
   5.0592513e+00  6.1475520e+00  6.3570652e+00  5.6636381e+00
   6.1225452e+00  5.7980156e+00  5.6745620e+00  5.0219235e+00
   4.5807090e+00  5.6214705e+00  5.3590655e+00  6.2158031e+00
   6.8514595e+00  6.1311817e+00  4.9394560e+00  5.7861238e+00
   5.2885504e+00  5.8875041e+00  5.7129755e+00  4.4113078e+00
   6.0710526e+00  5.2563438e+00  4.3051038e+00  5.9377909e+00
   5.3053098e+00  4.6715317e+00  4.4074697e+00  6.0630517e+00
   1.7303765e+00  4.5227230e-01  7.4014431e-01  5.7905692e-01
   2.6512140e-01 -2.6138017e-01  3.9958957e-01 -3.0835450e-02
   2.3187983e-01  5.1093531e-01  3.3035854e-01 -6.6076440e-01
   4.4085237e-01  8.7131631e-01  1.0041832e+00 -8.0556542e-01
   8.5022938e-01  6.0287613e-01 -3.8050425e-01 -1.7403919e-01
   8.1947899e-01 -2.2986948e-02  7.0710444e-01  1.4368406e-01
  -8.7472916e-02  5.6807011e-02  4.5708066e-01 -1.3038515e+00
   3.4293473e-02 -1.0111295e+00 -8.4716249e-01  5.2130079e-01
  -1.3285892e+00  1.2889528e-01 -1.3990200e-01  2.0161593e-01
   1.3920230e+00  2.9550713e-01 -5.4254901e-01  2.9656810e-01
  -1.2197701e+00  4.1106045e-01  3.6626405e-01  1.1589074e+00
   1.8694024e-01  5.5356401e-01  3.7459373e-01 -3.7928078e-01
  -7.9731762e-02  1.2072837e-01 -7.1227986e-01 -2.4660003e-01
  -7.4367091e-02  1.0144631e+00 -1.6049421e-01 -1.0977416e-01
  -7.7048886e-01  7.0240140e-01  1.9719699e-01  3.0759150e-01
  -3.0240282e-02 -2.0547447e-01  1.0990901e+00 -7.5426364e-01
  -3.2639343e-01 -3.5809979e-01  2.8601173e-01  4.4592258e-01
  -4.0671444e-01 -1.3729658e+00 -1.7300189e-02  5.0513840e-01
  -7.5249434e-01 -4.6411619e-01  6.9671625e-01  2.9343644e-01
   5.9735888e-01 -3.4530112e-01  9.9859089e-03 -5.4003698e-01
  -6.8783146e-01 -1.5815228e-01 -7.2321272e-01  6.8602681e-02
  -5.1300949e-01  7.4352551e-01  1.1527917e-01 -4.3437594e-01
   8.7715662e-01  5.4800314e-01  2.3377486e-01 -4.8172858e-01
   7.0777947e-01  2.7547348e-01 -9.3974274e-01  1.1464468e-01
   2.7037811e-01  7.7887481e-01  8.8227540e-03  9.7102636e-01
   1.7867106e-01 -6.6159159e-02  3.0611596e-01 -4.7162247e-01
  -1.5381978e+00 -1.0896037e+00 -4.7097319e-01 -1.1610851e+00
  -3.5476536e-01 -7.5060451e-01  7.2767842e-01 -1.7641962e-02
  -1.2753413e+00  1.1304152e+00 -1.2069564e+00 -6.4753646e-01
  -7.1320236e-01 -2.9475993e-01  8.6274344e-01 -2.7191919e-01
   4.5367360e-02  5.1542048e+00  6.1385674e+00  5.2652063e+00
   6.1101789e+00  5.5047774e+00  6.1414189e+00  5.3824792e+00
   4.9515414e+00  4.9475927e+00  6.9570150e+00  6.1509032e+00
   5.8201313e+00  6.7705474e+00  4.9619408e+00  4.6780720e+00
   7.1445484e+00  4.8484631e+00  5.5808148e+00  6.2492280e+00
   6.6638274e+00  5.4812665e+00  5.9779058e+00  6.4405313e+00
   5.3026605e+00  6.8636427e+00  5.2321610e+00  5.9709840e+00
   6.2595949e+00  5.7974911e+00  6.6715865e+00  5.8391442e+00
   7.0969582e+00  5.6488380e+00  4.8637986e+00  6.5416207e+00
   5.6321149e+00  5.6299858e+00  7.1962667e+00  6.3725343e+00
   5.2298284e+00  6.3585691e+00  5.2604084e+00  6.1074338e+00
   5.6279511e+00  5.3189163e+00  5.8129640e+00  6.3179779e+00
   6.8262305e+00  6.5368128e+00  6.7328916e+00  7.1370993e+00
   5.7869945e+00  6.5009627e+00  4.9952483e+00  6.1049314e+00
   5.9833302e+00  4.6856270e+00  6.8941674e+00  5.8601928e+00
   1.1501909e+00  1.3204268e+00  1.9565828e+00  9.5807010e-01
   3.8412917e-01  7.6953769e-01  1.1903036e+00  8.8815194e-01
   2.1008580e+00  4.6285272e-01  1.4223310e+00  7.5523555e-01
   6.6982508e-01  9.5008034e-01  8.9071149e-01  3.8548851e-01
   1.4508973e+00  2.1880155e+00  7.9208064e-01  1.1110572e+00
   7.4616081e-01  2.7013035e+00  1.2303709e+00  5.7448918e-01
   2.0801160e+00  9.9463004e-01  4.9605304e-01  1.9442124e+00
   6.1482114e-01  6.6225719e-01  5.7386333e-01  8.1500125e-01
   8.6347026e-01  1.3602996e+00  2.1278448e+00  1.1070150e+00
   2.2861052e-01  1.0135152e+00  6.4501143e-01  2.8218830e-01
   2.1263602e+00  4.8681784e-01  7.1539629e-01  4.2140573e-01
   9.5670444e-01  7.8698009e-01  1.8183081e+00  5.6292886e-01
   8.6978871e-01  5.3939039e-01  1.7876087e+00  5.8691812e-01
   2.0471973e+00  1.0313033e+00  1.5051231e+00  7.3934901e-01
   8.8473135e-01  3.9022470e-01  9.2453885e-01  3.8938558e-01
   2.2872586e+00  5.4373044e-01  1.1985613e+00  4.9930805e-01
   1.5309411e+00  2.0416272e+00  1.2629482e+00  1.0701822e+00
   6.2382764e-01  1.2147692e+00  4.4084859e-01  1.8254027e+00
   4.4548404e-01  8.2244796e-01  1.1251363e+00  6.6077489e-01
   1.3727627e+00  1.1571940e+00  7.5508422e-01  9.0624470e-01
   1.0765303e+00  2.0250573e+00  2.1349862e+00  1.1563851e+00
   1.4192418e+00  4.3398541e-01  5.4513067e-01  8.1730002e-01
   1.2764249e+00  1.2278619e+00  2.6351482e-01  8.7307107e-01
   4.5484138e-01  1.3191333e+00  1.1030290e+00  4.3288654e-01
   1.0900981e+00  8.7965548e-01  1.1900558e+00  6.3230366e-01
   1.3666199e+00  1.7730753e+00  7.7008450e-01  6.6335320e-01
   1.3576224e+00  1.0837702e+00  8.1901670e-01  2.8108275e-01
   7.0109129e-01  2.1298385e+00  2.5976276e-01  1.6727650e+00
   8.1833029e-01  2.0167875e+00  3.2689989e-01  1.6228776e+00
   1.2402512e+00  1.0307205e+00  1.3357641e+00  1.4038645e+00
   3.0572715e+00 -2.9771764e+00 -9.0909052e+00]]
  ### Calculate Metrics    ######################################## 
  date_run                              2020-05-09 10:50:48.717728
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   96.8454
metric_name                                  mean_absolute_error
Name: 0, dtype: object 
  date_run                              2020-05-09 10:50:48.722298
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9397.29
metric_name                                   mean_squared_error
Name: 1, dtype: object 
  date_run                              2020-05-09 10:50:48.730851
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   96.9181
metric_name                                median_absolute_error
Name: 2, dtype: object 
  date_run                              2020-05-09 10:50:48.734009
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -840.594
metric_name                                             r2_score
Name: 3, dtype: object 
  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256} 
  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139958555867176
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139957616626880
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139957616627384
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139957347598576
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139957347599080
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139957347599584
  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4ac2bb6400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.767964
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.715661
grad_step = 000002, loss = 0.670095
grad_step = 000003, loss = 0.619984
grad_step = 000004, loss = 0.563080
grad_step = 000005, loss = 0.509153
grad_step = 000006, loss = 0.477734
grad_step = 000007, loss = 0.462503
grad_step = 000008, loss = 0.442392
grad_step = 000009, loss = 0.408186
grad_step = 000010, loss = 0.377092
grad_step = 000011, loss = 0.360748
grad_step = 000012, loss = 0.351323
grad_step = 000013, loss = 0.339704
grad_step = 000014, loss = 0.324287
grad_step = 000015, loss = 0.307674
grad_step = 000016, loss = 0.292100
grad_step = 000017, loss = 0.278074
grad_step = 000018, loss = 0.264254
grad_step = 000019, loss = 0.250553
grad_step = 000020, loss = 0.237706
grad_step = 000021, loss = 0.225487
grad_step = 000022, loss = 0.214820
grad_step = 000023, loss = 0.205564
grad_step = 000024, loss = 0.195913
grad_step = 000025, loss = 0.184912
grad_step = 000026, loss = 0.173587
grad_step = 000027, loss = 0.163517
grad_step = 000028, loss = 0.155069
grad_step = 000029, loss = 0.147106
grad_step = 000030, loss = 0.138642
grad_step = 000031, loss = 0.130026
grad_step = 000032, loss = 0.122063
grad_step = 000033, loss = 0.114904
grad_step = 000034, loss = 0.108158
grad_step = 000035, loss = 0.101573
grad_step = 000036, loss = 0.095292
grad_step = 000037, loss = 0.089551
grad_step = 000038, loss = 0.084245
grad_step = 000039, loss = 0.078945
grad_step = 000040, loss = 0.073626
grad_step = 000041, loss = 0.068693
grad_step = 000042, loss = 0.064223
grad_step = 000043, loss = 0.060044
grad_step = 000044, loss = 0.056126
grad_step = 000045, loss = 0.052468
grad_step = 000046, loss = 0.048980
grad_step = 000047, loss = 0.045675
grad_step = 000048, loss = 0.042590
grad_step = 000049, loss = 0.039657
grad_step = 000050, loss = 0.036881
grad_step = 000051, loss = 0.034319
grad_step = 000052, loss = 0.031935
grad_step = 000053, loss = 0.029687
grad_step = 000054, loss = 0.027553
grad_step = 000055, loss = 0.025513
grad_step = 000056, loss = 0.023611
grad_step = 000057, loss = 0.021906
grad_step = 000058, loss = 0.020332
grad_step = 000059, loss = 0.018820
grad_step = 000060, loss = 0.017410
grad_step = 000061, loss = 0.016119
grad_step = 000062, loss = 0.014914
grad_step = 000063, loss = 0.013776
grad_step = 000064, loss = 0.012720
grad_step = 000065, loss = 0.011767
grad_step = 000066, loss = 0.010889
grad_step = 000067, loss = 0.010050
grad_step = 000068, loss = 0.009283
grad_step = 000069, loss = 0.008604
grad_step = 000070, loss = 0.007975
grad_step = 000071, loss = 0.007384
grad_step = 000072, loss = 0.006849
grad_step = 000073, loss = 0.006376
grad_step = 000074, loss = 0.005940
grad_step = 000075, loss = 0.005526
grad_step = 000076, loss = 0.005156
grad_step = 000077, loss = 0.004832
grad_step = 000078, loss = 0.004531
grad_step = 000079, loss = 0.004257
grad_step = 000080, loss = 0.004017
grad_step = 000081, loss = 0.003802
grad_step = 000082, loss = 0.003603
grad_step = 000083, loss = 0.003425
grad_step = 000084, loss = 0.003274
grad_step = 000085, loss = 0.003136
grad_step = 000086, loss = 0.003011
grad_step = 000087, loss = 0.002904
grad_step = 000088, loss = 0.002809
grad_step = 000089, loss = 0.002721
grad_step = 000090, loss = 0.002643
grad_step = 000091, loss = 0.002578
grad_step = 000092, loss = 0.002518
grad_step = 000093, loss = 0.002464
grad_step = 000094, loss = 0.002418
grad_step = 000095, loss = 0.002376
grad_step = 000096, loss = 0.002338
grad_step = 000097, loss = 0.002305
grad_step = 000098, loss = 0.002276
grad_step = 000099, loss = 0.002249
grad_step = 000100, loss = 0.002225
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002204
grad_step = 000102, loss = 0.002184
grad_step = 000103, loss = 0.002167
grad_step = 000104, loss = 0.002151
grad_step = 000105, loss = 0.002137
grad_step = 000106, loss = 0.002124
grad_step = 000107, loss = 0.002113
grad_step = 000108, loss = 0.002103
grad_step = 000109, loss = 0.002095
grad_step = 000110, loss = 0.002084
grad_step = 000111, loss = 0.002077
grad_step = 000112, loss = 0.002070
grad_step = 000113, loss = 0.002063
grad_step = 000114, loss = 0.002056
grad_step = 000115, loss = 0.002051
grad_step = 000116, loss = 0.002045
grad_step = 000117, loss = 0.002040
grad_step = 000118, loss = 0.002035
grad_step = 000119, loss = 0.002030
grad_step = 000120, loss = 0.002025
grad_step = 000121, loss = 0.002020
grad_step = 000122, loss = 0.002015
grad_step = 000123, loss = 0.002011
grad_step = 000124, loss = 0.002006
grad_step = 000125, loss = 0.002001
grad_step = 000126, loss = 0.001996
grad_step = 000127, loss = 0.001991
grad_step = 000128, loss = 0.001985
grad_step = 000129, loss = 0.001980
grad_step = 000130, loss = 0.001974
grad_step = 000131, loss = 0.001969
grad_step = 000132, loss = 0.001963
grad_step = 000133, loss = 0.001956
grad_step = 000134, loss = 0.001950
grad_step = 000135, loss = 0.001944
grad_step = 000136, loss = 0.001937
grad_step = 000137, loss = 0.001930
grad_step = 000138, loss = 0.001923
grad_step = 000139, loss = 0.001915
grad_step = 000140, loss = 0.001907
grad_step = 000141, loss = 0.001899
grad_step = 000142, loss = 0.001891
grad_step = 000143, loss = 0.001883
grad_step = 000144, loss = 0.001874
grad_step = 000145, loss = 0.001866
grad_step = 000146, loss = 0.001858
grad_step = 000147, loss = 0.001850
grad_step = 000148, loss = 0.001840
grad_step = 000149, loss = 0.001830
grad_step = 000150, loss = 0.001823
grad_step = 000151, loss = 0.001814
grad_step = 000152, loss = 0.001803
grad_step = 000153, loss = 0.001794
grad_step = 000154, loss = 0.001786
grad_step = 000155, loss = 0.001778
grad_step = 000156, loss = 0.001768
grad_step = 000157, loss = 0.001756
grad_step = 000158, loss = 0.001746
grad_step = 000159, loss = 0.001736
grad_step = 000160, loss = 0.001727
grad_step = 000161, loss = 0.001721
grad_step = 000162, loss = 0.001723
grad_step = 000163, loss = 0.001750
grad_step = 000164, loss = 0.001768
grad_step = 000165, loss = 0.001734
grad_step = 000166, loss = 0.001678
grad_step = 000167, loss = 0.001692
grad_step = 000168, loss = 0.001724
grad_step = 000169, loss = 0.001690
grad_step = 000170, loss = 0.001650
grad_step = 000171, loss = 0.001664
grad_step = 000172, loss = 0.001681
grad_step = 000173, loss = 0.001657
grad_step = 000174, loss = 0.001626
grad_step = 000175, loss = 0.001632
grad_step = 000176, loss = 0.001649
grad_step = 000177, loss = 0.001636
grad_step = 000178, loss = 0.001608
grad_step = 000179, loss = 0.001598
grad_step = 000180, loss = 0.001607
grad_step = 000181, loss = 0.001616
grad_step = 000182, loss = 0.001605
grad_step = 000183, loss = 0.001586
grad_step = 000184, loss = 0.001570
grad_step = 000185, loss = 0.001566
grad_step = 000186, loss = 0.001566
grad_step = 000187, loss = 0.001572
grad_step = 000188, loss = 0.001579
grad_step = 000189, loss = 0.001574
grad_step = 000190, loss = 0.001569
grad_step = 000191, loss = 0.001554
grad_step = 000192, loss = 0.001537
grad_step = 000193, loss = 0.001523
grad_step = 000194, loss = 0.001518
grad_step = 000195, loss = 0.001516
grad_step = 000196, loss = 0.001515
grad_step = 000197, loss = 0.001519
grad_step = 000198, loss = 0.001530
grad_step = 000199, loss = 0.001552
grad_step = 000200, loss = 0.001575
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001596
grad_step = 000202, loss = 0.001572
grad_step = 000203, loss = 0.001522
grad_step = 000204, loss = 0.001476
grad_step = 000205, loss = 0.001482
grad_step = 000206, loss = 0.001513
grad_step = 000207, loss = 0.001511
grad_step = 000208, loss = 0.001482
grad_step = 000209, loss = 0.001462
grad_step = 000210, loss = 0.001465
grad_step = 000211, loss = 0.001471
grad_step = 000212, loss = 0.001467
grad_step = 000213, loss = 0.001462
grad_step = 000214, loss = 0.001451
grad_step = 000215, loss = 0.001437
grad_step = 000216, loss = 0.001438
grad_step = 000217, loss = 0.001447
grad_step = 000218, loss = 0.001443
grad_step = 000219, loss = 0.001427
grad_step = 000220, loss = 0.001420
grad_step = 000221, loss = 0.001421
grad_step = 000222, loss = 0.001419
grad_step = 000223, loss = 0.001415
grad_step = 000224, loss = 0.001415
grad_step = 000225, loss = 0.001414
grad_step = 000226, loss = 0.001406
grad_step = 000227, loss = 0.001398
grad_step = 000228, loss = 0.001394
grad_step = 000229, loss = 0.001393
grad_step = 000230, loss = 0.001389
grad_step = 000231, loss = 0.001384
grad_step = 000232, loss = 0.001380
grad_step = 000233, loss = 0.001379
grad_step = 000234, loss = 0.001380
grad_step = 000235, loss = 0.001381
grad_step = 000236, loss = 0.001384
grad_step = 000237, loss = 0.001393
grad_step = 000238, loss = 0.001417
grad_step = 000239, loss = 0.001453
grad_step = 000240, loss = 0.001503
grad_step = 000241, loss = 0.001506
grad_step = 000242, loss = 0.001478
grad_step = 000243, loss = 0.001399
grad_step = 000244, loss = 0.001350
grad_step = 000245, loss = 0.001362
grad_step = 000246, loss = 0.001401
grad_step = 000247, loss = 0.001407
grad_step = 000248, loss = 0.001360
grad_step = 000249, loss = 0.001329
grad_step = 000250, loss = 0.001345
grad_step = 000251, loss = 0.001367
grad_step = 000252, loss = 0.001356
grad_step = 000253, loss = 0.001323
grad_step = 000254, loss = 0.001315
grad_step = 000255, loss = 0.001331
grad_step = 000256, loss = 0.001336
grad_step = 000257, loss = 0.001323
grad_step = 000258, loss = 0.001305
grad_step = 000259, loss = 0.001303
grad_step = 000260, loss = 0.001307
grad_step = 000261, loss = 0.001304
grad_step = 000262, loss = 0.001297
grad_step = 000263, loss = 0.001294
grad_step = 000264, loss = 0.001295
grad_step = 000265, loss = 0.001292
grad_step = 000266, loss = 0.001283
grad_step = 000267, loss = 0.001274
grad_step = 000268, loss = 0.001269
grad_step = 000269, loss = 0.001269
grad_step = 000270, loss = 0.001269
grad_step = 000271, loss = 0.001266
grad_step = 000272, loss = 0.001264
grad_step = 000273, loss = 0.001267
grad_step = 000274, loss = 0.001280
grad_step = 000275, loss = 0.001318
grad_step = 000276, loss = 0.001379
grad_step = 000277, loss = 0.001494
grad_step = 000278, loss = 0.001517
grad_step = 000279, loss = 0.001474
grad_step = 000280, loss = 0.001297
grad_step = 000281, loss = 0.001245
grad_step = 000282, loss = 0.001333
grad_step = 000283, loss = 0.001372
grad_step = 000284, loss = 0.001318
grad_step = 000285, loss = 0.001238
grad_step = 000286, loss = 0.001268
grad_step = 000287, loss = 0.001334
grad_step = 000288, loss = 0.001297
grad_step = 000289, loss = 0.001236
grad_step = 000290, loss = 0.001233
grad_step = 000291, loss = 0.001270
grad_step = 000292, loss = 0.001279
grad_step = 000293, loss = 0.001237
grad_step = 000294, loss = 0.001213
grad_step = 000295, loss = 0.001235
grad_step = 000296, loss = 0.001255
grad_step = 000297, loss = 0.001242
grad_step = 000298, loss = 0.001210
grad_step = 000299, loss = 0.001205
grad_step = 000300, loss = 0.001220
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001228
grad_step = 000302, loss = 0.001221
grad_step = 000303, loss = 0.001201
grad_step = 000304, loss = 0.001193
grad_step = 000305, loss = 0.001200
grad_step = 000306, loss = 0.001209
grad_step = 000307, loss = 0.001210
grad_step = 000308, loss = 0.001200
grad_step = 000309, loss = 0.001189
grad_step = 000310, loss = 0.001182
grad_step = 000311, loss = 0.001181
grad_step = 000312, loss = 0.001185
grad_step = 000313, loss = 0.001189
grad_step = 000314, loss = 0.001191
grad_step = 000315, loss = 0.001189
grad_step = 000316, loss = 0.001185
grad_step = 000317, loss = 0.001179
grad_step = 000318, loss = 0.001174
grad_step = 000319, loss = 0.001171
grad_step = 000320, loss = 0.001168
grad_step = 000321, loss = 0.001167
grad_step = 000322, loss = 0.001168
grad_step = 000323, loss = 0.001171
grad_step = 000324, loss = 0.001178
grad_step = 000325, loss = 0.001192
grad_step = 000326, loss = 0.001217
grad_step = 000327, loss = 0.001253
grad_step = 000328, loss = 0.001293
grad_step = 000329, loss = 0.001320
grad_step = 000330, loss = 0.001295
grad_step = 000331, loss = 0.001232
grad_step = 000332, loss = 0.001168
grad_step = 000333, loss = 0.001160
grad_step = 000334, loss = 0.001198
grad_step = 000335, loss = 0.001230
grad_step = 000336, loss = 0.001211
grad_step = 000337, loss = 0.001169
grad_step = 000338, loss = 0.001148
grad_step = 000339, loss = 0.001164
grad_step = 000340, loss = 0.001187
grad_step = 000341, loss = 0.001192
grad_step = 000342, loss = 0.001175
grad_step = 000343, loss = 0.001156
grad_step = 000344, loss = 0.001150
grad_step = 000345, loss = 0.001165
grad_step = 000346, loss = 0.001189
grad_step = 000347, loss = 0.001214
grad_step = 000348, loss = 0.001226
grad_step = 000349, loss = 0.001251
grad_step = 000350, loss = 0.001288
grad_step = 000351, loss = 0.001376
grad_step = 000352, loss = 0.001398
grad_step = 000353, loss = 0.001358
grad_step = 000354, loss = 0.001200
grad_step = 000355, loss = 0.001136
grad_step = 000356, loss = 0.001205
grad_step = 000357, loss = 0.001242
grad_step = 000358, loss = 0.001189
grad_step = 000359, loss = 0.001138
grad_step = 000360, loss = 0.001177
grad_step = 000361, loss = 0.001204
grad_step = 000362, loss = 0.001156
grad_step = 000363, loss = 0.001130
grad_step = 000364, loss = 0.001163
grad_step = 000365, loss = 0.001177
grad_step = 000366, loss = 0.001144
grad_step = 000367, loss = 0.001122
grad_step = 000368, loss = 0.001141
grad_step = 000369, loss = 0.001162
grad_step = 000370, loss = 0.001146
grad_step = 000371, loss = 0.001122
grad_step = 000372, loss = 0.001118
grad_step = 000373, loss = 0.001133
grad_step = 000374, loss = 0.001143
grad_step = 000375, loss = 0.001134
grad_step = 000376, loss = 0.001118
grad_step = 000377, loss = 0.001113
grad_step = 000378, loss = 0.001120
grad_step = 000379, loss = 0.001127
grad_step = 000380, loss = 0.001123
grad_step = 000381, loss = 0.001114
grad_step = 000382, loss = 0.001108
grad_step = 000383, loss = 0.001108
grad_step = 000384, loss = 0.001112
grad_step = 000385, loss = 0.001114
grad_step = 000386, loss = 0.001110
grad_step = 000387, loss = 0.001105
grad_step = 000388, loss = 0.001104
grad_step = 000389, loss = 0.001107
grad_step = 000390, loss = 0.001113
grad_step = 000391, loss = 0.001118
grad_step = 000392, loss = 0.001127
grad_step = 000393, loss = 0.001143
grad_step = 000394, loss = 0.001179
grad_step = 000395, loss = 0.001233
grad_step = 000396, loss = 0.001327
grad_step = 000397, loss = 0.001387
grad_step = 000398, loss = 0.001433
grad_step = 000399, loss = 0.001312
grad_step = 000400, loss = 0.001166
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001095
grad_step = 000402, loss = 0.001148
grad_step = 000403, loss = 0.001227
grad_step = 000404, loss = 0.001197
grad_step = 000405, loss = 0.001123
grad_step = 000406, loss = 0.001099
grad_step = 000407, loss = 0.001141
grad_step = 000408, loss = 0.001171
grad_step = 000409, loss = 0.001134
grad_step = 000410, loss = 0.001100
grad_step = 000411, loss = 0.001104
grad_step = 000412, loss = 0.001124
grad_step = 000413, loss = 0.001122
grad_step = 000414, loss = 0.001099
grad_step = 000415, loss = 0.001091
grad_step = 000416, loss = 0.001100
grad_step = 000417, loss = 0.001105
grad_step = 000418, loss = 0.001098
grad_step = 000419, loss = 0.001087
grad_step = 000420, loss = 0.001085
grad_step = 000421, loss = 0.001088
grad_step = 000422, loss = 0.001088
grad_step = 000423, loss = 0.001083
grad_step = 000424, loss = 0.001078
grad_step = 000425, loss = 0.001079
grad_step = 000426, loss = 0.001081
grad_step = 000427, loss = 0.001080
grad_step = 000428, loss = 0.001074
grad_step = 000429, loss = 0.001069
grad_step = 000430, loss = 0.001067
grad_step = 000431, loss = 0.001069
grad_step = 000432, loss = 0.001073
grad_step = 000433, loss = 0.001073
grad_step = 000434, loss = 0.001071
grad_step = 000435, loss = 0.001066
grad_step = 000436, loss = 0.001063
grad_step = 000437, loss = 0.001063
grad_step = 000438, loss = 0.001063
grad_step = 000439, loss = 0.001062
grad_step = 000440, loss = 0.001060
grad_step = 000441, loss = 0.001058
grad_step = 000442, loss = 0.001055
grad_step = 000443, loss = 0.001054
grad_step = 000444, loss = 0.001054
grad_step = 000445, loss = 0.001054
grad_step = 000446, loss = 0.001054
grad_step = 000447, loss = 0.001053
grad_step = 000448, loss = 0.001052
grad_step = 000449, loss = 0.001051
grad_step = 000450, loss = 0.001052
grad_step = 000451, loss = 0.001053
grad_step = 000452, loss = 0.001057
grad_step = 000453, loss = 0.001065
grad_step = 000454, loss = 0.001077
grad_step = 000455, loss = 0.001103
grad_step = 000456, loss = 0.001143
grad_step = 000457, loss = 0.001222
grad_step = 000458, loss = 0.001306
grad_step = 000459, loss = 0.001441
grad_step = 000460, loss = 0.001422
grad_step = 000461, loss = 0.001342
grad_step = 000462, loss = 0.001157
grad_step = 000463, loss = 0.001075
grad_step = 000464, loss = 0.001113
grad_step = 000465, loss = 0.001165
grad_step = 000466, loss = 0.001164
grad_step = 000467, loss = 0.001099
grad_step = 000468, loss = 0.001079
grad_step = 000469, loss = 0.001103
grad_step = 000470, loss = 0.001105
grad_step = 000471, loss = 0.001086
grad_step = 000472, loss = 0.001069
grad_step = 000473, loss = 0.001073
grad_step = 000474, loss = 0.001082
grad_step = 000475, loss = 0.001072
grad_step = 000476, loss = 0.001060
grad_step = 000477, loss = 0.001052
grad_step = 000478, loss = 0.001055
grad_step = 000479, loss = 0.001062
grad_step = 000480, loss = 0.001058
grad_step = 000481, loss = 0.001048
grad_step = 000482, loss = 0.001037
grad_step = 000483, loss = 0.001034
grad_step = 000484, loss = 0.001042
grad_step = 000485, loss = 0.001047
grad_step = 000486, loss = 0.001045
grad_step = 000487, loss = 0.001032
grad_step = 000488, loss = 0.001022
grad_step = 000489, loss = 0.001021
grad_step = 000490, loss = 0.001028
grad_step = 000491, loss = 0.001035
grad_step = 000492, loss = 0.001035
grad_step = 000493, loss = 0.001031
grad_step = 000494, loss = 0.001023
grad_step = 000495, loss = 0.001017
grad_step = 000496, loss = 0.001016
grad_step = 000497, loss = 0.001016
grad_step = 000498, loss = 0.001016
grad_step = 000499, loss = 0.001015
grad_step = 000500, loss = 0.001015
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001015
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
  date_run                              2020-05-09 10:51:11.605449
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.238085
metric_name                                  mean_absolute_error
Name: 4, dtype: object 
  date_run                              2020-05-09 10:51:11.610994
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.137738
metric_name                                   mean_squared_error
Name: 5, dtype: object 
  date_run                              2020-05-09 10:51:11.617898
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.139456
metric_name                                median_absolute_error
Name: 6, dtype: object 
  date_run                              2020-05-09 10:51:11.623194
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.09298
metric_name                                             r2_score
Name: 7, dtype: object 
  


### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}} 
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}} 
  #### Setup Model   ############################################## 

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range 
  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/ 
                       date_run  ...            metric_name
0  2020-05-09 10:50:48.717728  ...    mean_absolute_error
1  2020-05-09 10:50:48.722298  ...     mean_squared_error
2  2020-05-09 10:50:48.730851  ...  median_absolute_error
3  2020-05-09 10:50:48.734009  ...               r2_score
4  2020-05-09 10:51:11.605449  ...    mean_absolute_error
5  2020-05-09 10:51:11.610994  ...     mean_squared_error
6  2020-05-09 10:51:11.617898  ...  median_absolute_error
7  2020-05-09 10:51:11.623194  ...               r2_score

[8 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 117, in benchmark_run
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
  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256} 
  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139946300318384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139945021443936
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139945020993944
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139945020994448
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139945020994952
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139945020995456
  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f47d423f358> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.660782
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.621104
grad_step = 000002, loss = 0.589665
grad_step = 000003, loss = 0.556165
grad_step = 000004, loss = 0.517628
grad_step = 000005, loss = 0.476177
grad_step = 000006, loss = 0.442211
grad_step = 000007, loss = 0.432228
grad_step = 000008, loss = 0.430507
grad_step = 000009, loss = 0.410646
grad_step = 000010, loss = 0.382885
grad_step = 000011, loss = 0.362524
grad_step = 000012, loss = 0.350899
grad_step = 000013, loss = 0.342639
grad_step = 000014, loss = 0.333350
grad_step = 000015, loss = 0.322527
grad_step = 000016, loss = 0.311043
grad_step = 000017, loss = 0.299256
grad_step = 000018, loss = 0.286913
grad_step = 000019, loss = 0.274201
grad_step = 000020, loss = 0.262814
grad_step = 000021, loss = 0.253743
grad_step = 000022, loss = 0.244869
grad_step = 000023, loss = 0.234323
grad_step = 000024, loss = 0.223494
grad_step = 000025, loss = 0.214140
grad_step = 000026, loss = 0.205895
grad_step = 000027, loss = 0.197833
grad_step = 000028, loss = 0.189598
grad_step = 000029, loss = 0.181313
grad_step = 000030, loss = 0.173158
grad_step = 000031, loss = 0.165156
grad_step = 000032, loss = 0.157359
grad_step = 000033, loss = 0.150028
grad_step = 000034, loss = 0.143321
grad_step = 000035, loss = 0.136808
grad_step = 000036, loss = 0.130161
grad_step = 000037, loss = 0.123643
grad_step = 000038, loss = 0.117372
grad_step = 000039, loss = 0.111203
grad_step = 000040, loss = 0.105244
grad_step = 000041, loss = 0.099744
grad_step = 000042, loss = 0.094654
grad_step = 000043, loss = 0.089661
grad_step = 000044, loss = 0.084642
grad_step = 000045, loss = 0.079775
grad_step = 000046, loss = 0.075175
grad_step = 000047, loss = 0.070876
grad_step = 000048, loss = 0.066860
grad_step = 000049, loss = 0.062981
grad_step = 000050, loss = 0.059157
grad_step = 000051, loss = 0.055547
grad_step = 000052, loss = 0.052200
grad_step = 000053, loss = 0.048974
grad_step = 000054, loss = 0.045921
grad_step = 000055, loss = 0.043089
grad_step = 000056, loss = 0.040383
grad_step = 000057, loss = 0.037827
grad_step = 000058, loss = 0.035450
grad_step = 000059, loss = 0.033199
grad_step = 000060, loss = 0.031108
grad_step = 000061, loss = 0.029158
grad_step = 000062, loss = 0.027292
grad_step = 000063, loss = 0.025558
grad_step = 000064, loss = 0.023946
grad_step = 000065, loss = 0.022450
grad_step = 000066, loss = 0.021058
grad_step = 000067, loss = 0.019721
grad_step = 000068, loss = 0.018492
grad_step = 000069, loss = 0.017370
grad_step = 000070, loss = 0.016319
grad_step = 000071, loss = 0.015337
grad_step = 000072, loss = 0.014411
grad_step = 000073, loss = 0.013559
grad_step = 000074, loss = 0.012765
grad_step = 000075, loss = 0.012031
grad_step = 000076, loss = 0.011349
grad_step = 000077, loss = 0.010703
grad_step = 000078, loss = 0.010102
grad_step = 000079, loss = 0.009538
grad_step = 000080, loss = 0.009024
grad_step = 000081, loss = 0.008538
grad_step = 000082, loss = 0.008081
grad_step = 000083, loss = 0.007649
grad_step = 000084, loss = 0.007251
grad_step = 000085, loss = 0.006875
grad_step = 000086, loss = 0.006525
grad_step = 000087, loss = 0.006195
grad_step = 000088, loss = 0.005885
grad_step = 000089, loss = 0.005591
grad_step = 000090, loss = 0.005319
grad_step = 000091, loss = 0.005065
grad_step = 000092, loss = 0.004826
grad_step = 000093, loss = 0.004597
grad_step = 000094, loss = 0.004387
grad_step = 000095, loss = 0.004191
grad_step = 000096, loss = 0.004009
grad_step = 000097, loss = 0.003838
grad_step = 000098, loss = 0.003679
grad_step = 000099, loss = 0.003532
grad_step = 000100, loss = 0.003395
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003269
grad_step = 000102, loss = 0.003151
grad_step = 000103, loss = 0.003042
grad_step = 000104, loss = 0.002942
grad_step = 000105, loss = 0.002849
grad_step = 000106, loss = 0.002764
grad_step = 000107, loss = 0.002685
grad_step = 000108, loss = 0.002612
grad_step = 000109, loss = 0.002546
grad_step = 000110, loss = 0.002484
grad_step = 000111, loss = 0.002428
grad_step = 000112, loss = 0.002376
grad_step = 000113, loss = 0.002329
grad_step = 000114, loss = 0.002285
grad_step = 000115, loss = 0.002245
grad_step = 000116, loss = 0.002208
grad_step = 000117, loss = 0.002174
grad_step = 000118, loss = 0.002143
grad_step = 000119, loss = 0.002115
grad_step = 000120, loss = 0.002088
grad_step = 000121, loss = 0.002065
grad_step = 000122, loss = 0.002042
grad_step = 000123, loss = 0.002022
grad_step = 000124, loss = 0.002003
grad_step = 000125, loss = 0.001987
grad_step = 000126, loss = 0.001972
grad_step = 000127, loss = 0.001958
grad_step = 000128, loss = 0.001943
grad_step = 000129, loss = 0.001928
grad_step = 000130, loss = 0.001917
grad_step = 000131, loss = 0.001907
grad_step = 000132, loss = 0.001897
grad_step = 000133, loss = 0.001887
grad_step = 000134, loss = 0.001877
grad_step = 000135, loss = 0.001868
grad_step = 000136, loss = 0.001861
grad_step = 000137, loss = 0.001854
grad_step = 000138, loss = 0.001847
grad_step = 000139, loss = 0.001839
grad_step = 000140, loss = 0.001832
grad_step = 000141, loss = 0.001825
grad_step = 000142, loss = 0.001819
grad_step = 000143, loss = 0.001813
grad_step = 000144, loss = 0.001809
grad_step = 000145, loss = 0.001804
grad_step = 000146, loss = 0.001801
grad_step = 000147, loss = 0.001798
grad_step = 000148, loss = 0.001795
grad_step = 000149, loss = 0.001791
grad_step = 000150, loss = 0.001784
grad_step = 000151, loss = 0.001776
grad_step = 000152, loss = 0.001770
grad_step = 000153, loss = 0.001765
grad_step = 000154, loss = 0.001762
grad_step = 000155, loss = 0.001761
grad_step = 000156, loss = 0.001760
grad_step = 000157, loss = 0.001761
grad_step = 000158, loss = 0.001762
grad_step = 000159, loss = 0.001761
grad_step = 000160, loss = 0.001757
grad_step = 000161, loss = 0.001749
grad_step = 000162, loss = 0.001739
grad_step = 000163, loss = 0.001731
grad_step = 000164, loss = 0.001725
grad_step = 000165, loss = 0.001723
grad_step = 000166, loss = 0.001722
grad_step = 000167, loss = 0.001724
grad_step = 000168, loss = 0.001729
grad_step = 000169, loss = 0.001739
grad_step = 000170, loss = 0.001747
grad_step = 000171, loss = 0.001753
grad_step = 000172, loss = 0.001748
grad_step = 000173, loss = 0.001735
grad_step = 000174, loss = 0.001717
grad_step = 000175, loss = 0.001701
grad_step = 000176, loss = 0.001693
grad_step = 000177, loss = 0.001691
grad_step = 000178, loss = 0.001694
grad_step = 000179, loss = 0.001700
grad_step = 000180, loss = 0.001704
grad_step = 000181, loss = 0.001699
grad_step = 000182, loss = 0.001693
grad_step = 000183, loss = 0.001691
grad_step = 000184, loss = 0.001687
grad_step = 000185, loss = 0.001681
grad_step = 000186, loss = 0.001675
grad_step = 000187, loss = 0.001668
grad_step = 000188, loss = 0.001660
grad_step = 000189, loss = 0.001656
grad_step = 000190, loss = 0.001654
grad_step = 000191, loss = 0.001652
grad_step = 000192, loss = 0.001651
grad_step = 000193, loss = 0.001652
grad_step = 000194, loss = 0.001653
grad_step = 000195, loss = 0.001655
grad_step = 000196, loss = 0.001658
grad_step = 000197, loss = 0.001665
grad_step = 000198, loss = 0.001672
grad_step = 000199, loss = 0.001684
grad_step = 000200, loss = 0.001695
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001704
grad_step = 000202, loss = 0.001701
grad_step = 000203, loss = 0.001685
grad_step = 000204, loss = 0.001657
grad_step = 000205, loss = 0.001628
grad_step = 000206, loss = 0.001608
grad_step = 000207, loss = 0.001602
grad_step = 000208, loss = 0.001608
grad_step = 000209, loss = 0.001618
grad_step = 000210, loss = 0.001628
grad_step = 000211, loss = 0.001634
grad_step = 000212, loss = 0.001631
grad_step = 000213, loss = 0.001621
grad_step = 000214, loss = 0.001607
grad_step = 000215, loss = 0.001591
grad_step = 000216, loss = 0.001579
grad_step = 000217, loss = 0.001570
grad_step = 000218, loss = 0.001567
grad_step = 000219, loss = 0.001567
grad_step = 000220, loss = 0.001568
grad_step = 000221, loss = 0.001570
grad_step = 000222, loss = 0.001575
grad_step = 000223, loss = 0.001586
grad_step = 000224, loss = 0.001607
grad_step = 000225, loss = 0.001632
grad_step = 000226, loss = 0.001658
grad_step = 000227, loss = 0.001678
grad_step = 000228, loss = 0.001688
grad_step = 000229, loss = 0.001679
grad_step = 000230, loss = 0.001629
grad_step = 000231, loss = 0.001569
grad_step = 000232, loss = 0.001532
grad_step = 000233, loss = 0.001531
grad_step = 000234, loss = 0.001546
grad_step = 000235, loss = 0.001556
grad_step = 000236, loss = 0.001559
grad_step = 000237, loss = 0.001568
grad_step = 000238, loss = 0.001576
grad_step = 000239, loss = 0.001576
grad_step = 000240, loss = 0.001529
grad_step = 000241, loss = 0.001503
grad_step = 000242, loss = 0.001507
grad_step = 000243, loss = 0.001520
grad_step = 000244, loss = 0.001523
grad_step = 000245, loss = 0.001515
grad_step = 000246, loss = 0.001516
grad_step = 000247, loss = 0.001521
grad_step = 000248, loss = 0.001516
grad_step = 000249, loss = 0.001499
grad_step = 000250, loss = 0.001481
grad_step = 000251, loss = 0.001474
grad_step = 000252, loss = 0.001478
grad_step = 000253, loss = 0.001484
grad_step = 000254, loss = 0.001485
grad_step = 000255, loss = 0.001478
grad_step = 000256, loss = 0.001470
grad_step = 000257, loss = 0.001465
grad_step = 000258, loss = 0.001469
grad_step = 000259, loss = 0.001486
grad_step = 000260, loss = 0.001525
grad_step = 000261, loss = 0.001599
grad_step = 000262, loss = 0.001738
grad_step = 000263, loss = 0.001776
grad_step = 000264, loss = 0.001780
grad_step = 000265, loss = 0.001700
grad_step = 000266, loss = 0.001601
grad_step = 000267, loss = 0.001477
grad_step = 000268, loss = 0.001466
grad_step = 000269, loss = 0.001581
grad_step = 000270, loss = 0.001615
grad_step = 000271, loss = 0.001512
grad_step = 000272, loss = 0.001441
grad_step = 000273, loss = 0.001486
grad_step = 000274, loss = 0.001530
grad_step = 000275, loss = 0.001486
grad_step = 000276, loss = 0.001436
grad_step = 000277, loss = 0.001462
grad_step = 000278, loss = 0.001498
grad_step = 000279, loss = 0.001471
grad_step = 000280, loss = 0.001423
grad_step = 000281, loss = 0.001432
grad_step = 000282, loss = 0.001469
grad_step = 000283, loss = 0.001457
grad_step = 000284, loss = 0.001419
grad_step = 000285, loss = 0.001414
grad_step = 000286, loss = 0.001440
grad_step = 000287, loss = 0.001439
grad_step = 000288, loss = 0.001414
grad_step = 000289, loss = 0.001404
grad_step = 000290, loss = 0.001418
grad_step = 000291, loss = 0.001425
grad_step = 000292, loss = 0.001410
grad_step = 000293, loss = 0.001395
grad_step = 000294, loss = 0.001400
grad_step = 000295, loss = 0.001408
grad_step = 000296, loss = 0.001406
grad_step = 000297, loss = 0.001394
grad_step = 000298, loss = 0.001387
grad_step = 000299, loss = 0.001390
grad_step = 000300, loss = 0.001394
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001392
grad_step = 000302, loss = 0.001385
grad_step = 000303, loss = 0.001379
grad_step = 000304, loss = 0.001378
grad_step = 000305, loss = 0.001381
grad_step = 000306, loss = 0.001381
grad_step = 000307, loss = 0.001379
grad_step = 000308, loss = 0.001375
grad_step = 000309, loss = 0.001373
grad_step = 000310, loss = 0.001378
grad_step = 000311, loss = 0.001393
grad_step = 000312, loss = 0.001427
grad_step = 000313, loss = 0.001507
grad_step = 000314, loss = 0.001625
grad_step = 000315, loss = 0.001833
grad_step = 000316, loss = 0.002015
grad_step = 000317, loss = 0.002090
grad_step = 000318, loss = 0.001932
grad_step = 000319, loss = 0.001579
grad_step = 000320, loss = 0.001385
grad_step = 000321, loss = 0.001576
grad_step = 000322, loss = 0.001772
grad_step = 000323, loss = 0.001587
grad_step = 000324, loss = 0.001360
grad_step = 000325, loss = 0.001474
grad_step = 000326, loss = 0.001609
grad_step = 000327, loss = 0.001502
grad_step = 000328, loss = 0.001383
grad_step = 000329, loss = 0.001439
grad_step = 000330, loss = 0.001482
grad_step = 000331, loss = 0.001436
grad_step = 000332, loss = 0.001462
grad_step = 000333, loss = 0.001372
grad_step = 000334, loss = 0.001352
grad_step = 000335, loss = 0.001396
grad_step = 000336, loss = 0.001382
grad_step = 000337, loss = 0.001344
grad_step = 000338, loss = 0.001349
grad_step = 000339, loss = 0.001362
grad_step = 000340, loss = 0.001363
grad_step = 000341, loss = 0.001332
grad_step = 000342, loss = 0.001337
grad_step = 000343, loss = 0.001346
grad_step = 000344, loss = 0.001341
grad_step = 000345, loss = 0.001324
grad_step = 000346, loss = 0.001327
grad_step = 000347, loss = 0.001328
grad_step = 000348, loss = 0.001326
grad_step = 000349, loss = 0.001317
grad_step = 000350, loss = 0.001318
grad_step = 000351, loss = 0.001315
grad_step = 000352, loss = 0.001313
grad_step = 000353, loss = 0.001308
grad_step = 000354, loss = 0.001307
grad_step = 000355, loss = 0.001304
grad_step = 000356, loss = 0.001301
grad_step = 000357, loss = 0.001297
grad_step = 000358, loss = 0.001297
grad_step = 000359, loss = 0.001294
grad_step = 000360, loss = 0.001292
grad_step = 000361, loss = 0.001287
grad_step = 000362, loss = 0.001284
grad_step = 000363, loss = 0.001283
grad_step = 000364, loss = 0.001281
grad_step = 000365, loss = 0.001278
grad_step = 000366, loss = 0.001274
grad_step = 000367, loss = 0.001271
grad_step = 000368, loss = 0.001269
grad_step = 000369, loss = 0.001266
grad_step = 000370, loss = 0.001265
grad_step = 000371, loss = 0.001262
grad_step = 000372, loss = 0.001259
grad_step = 000373, loss = 0.001255
grad_step = 000374, loss = 0.001252
grad_step = 000375, loss = 0.001248
grad_step = 000376, loss = 0.001245
grad_step = 000377, loss = 0.001243
grad_step = 000378, loss = 0.001240
grad_step = 000379, loss = 0.001237
grad_step = 000380, loss = 0.001234
grad_step = 000381, loss = 0.001231
grad_step = 000382, loss = 0.001228
grad_step = 000383, loss = 0.001225
grad_step = 000384, loss = 0.001225
grad_step = 000385, loss = 0.001227
grad_step = 000386, loss = 0.001237
grad_step = 000387, loss = 0.001263
grad_step = 000388, loss = 0.001323
grad_step = 000389, loss = 0.001429
grad_step = 000390, loss = 0.001605
grad_step = 000391, loss = 0.001742
grad_step = 000392, loss = 0.001743
grad_step = 000393, loss = 0.001587
grad_step = 000394, loss = 0.001410
grad_step = 000395, loss = 0.001323
grad_step = 000396, loss = 0.001308
grad_step = 000397, loss = 0.001372
grad_step = 000398, loss = 0.001426
grad_step = 000399, loss = 0.001328
grad_step = 000400, loss = 0.001199
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001268
grad_step = 000402, loss = 0.001359
grad_step = 000403, loss = 0.001274
grad_step = 000404, loss = 0.001194
grad_step = 000405, loss = 0.001239
grad_step = 000406, loss = 0.001268
grad_step = 000407, loss = 0.001237
grad_step = 000408, loss = 0.001213
grad_step = 000409, loss = 0.001204
grad_step = 000410, loss = 0.001213
grad_step = 000411, loss = 0.001224
grad_step = 000412, loss = 0.001207
grad_step = 000413, loss = 0.001180
grad_step = 000414, loss = 0.001189
grad_step = 000415, loss = 0.001208
grad_step = 000416, loss = 0.001190
grad_step = 000417, loss = 0.001167
grad_step = 000418, loss = 0.001177
grad_step = 000419, loss = 0.001191
grad_step = 000420, loss = 0.001179
grad_step = 000421, loss = 0.001159
grad_step = 000422, loss = 0.001162
grad_step = 000423, loss = 0.001176
grad_step = 000424, loss = 0.001171
grad_step = 000425, loss = 0.001157
grad_step = 000426, loss = 0.001155
grad_step = 000427, loss = 0.001162
grad_step = 000428, loss = 0.001162
grad_step = 000429, loss = 0.001152
grad_step = 000430, loss = 0.001147
grad_step = 000431, loss = 0.001151
grad_step = 000432, loss = 0.001154
grad_step = 000433, loss = 0.001150
grad_step = 000434, loss = 0.001144
grad_step = 000435, loss = 0.001143
grad_step = 000436, loss = 0.001145
grad_step = 000437, loss = 0.001145
grad_step = 000438, loss = 0.001141
grad_step = 000439, loss = 0.001137
grad_step = 000440, loss = 0.001137
grad_step = 000441, loss = 0.001138
grad_step = 000442, loss = 0.001139
grad_step = 000443, loss = 0.001138
grad_step = 000444, loss = 0.001138
grad_step = 000445, loss = 0.001141
grad_step = 000446, loss = 0.001146
grad_step = 000447, loss = 0.001153
grad_step = 000448, loss = 0.001163
grad_step = 000449, loss = 0.001174
grad_step = 000450, loss = 0.001187
grad_step = 000451, loss = 0.001190
grad_step = 000452, loss = 0.001188
grad_step = 000453, loss = 0.001182
grad_step = 000454, loss = 0.001175
grad_step = 000455, loss = 0.001170
grad_step = 000456, loss = 0.001151
grad_step = 000457, loss = 0.001131
grad_step = 000458, loss = 0.001123
grad_step = 000459, loss = 0.001131
grad_step = 000460, loss = 0.001140
grad_step = 000461, loss = 0.001141
grad_step = 000462, loss = 0.001142
grad_step = 000463, loss = 0.001144
grad_step = 000464, loss = 0.001146
grad_step = 000465, loss = 0.001143
grad_step = 000466, loss = 0.001135
grad_step = 000467, loss = 0.001126
grad_step = 000468, loss = 0.001118
grad_step = 000469, loss = 0.001115
grad_step = 000470, loss = 0.001114
grad_step = 000471, loss = 0.001114
grad_step = 000472, loss = 0.001116
grad_step = 000473, loss = 0.001117
grad_step = 000474, loss = 0.001119
grad_step = 000475, loss = 0.001123
grad_step = 000476, loss = 0.001127
grad_step = 000477, loss = 0.001133
grad_step = 000478, loss = 0.001138
grad_step = 000479, loss = 0.001145
grad_step = 000480, loss = 0.001150
grad_step = 000481, loss = 0.001154
grad_step = 000482, loss = 0.001152
grad_step = 000483, loss = 0.001147
grad_step = 000484, loss = 0.001137
grad_step = 000485, loss = 0.001124
grad_step = 000486, loss = 0.001112
grad_step = 000487, loss = 0.001104
grad_step = 000488, loss = 0.001100
grad_step = 000489, loss = 0.001100
grad_step = 000490, loss = 0.001104
grad_step = 000491, loss = 0.001109
grad_step = 000492, loss = 0.001114
grad_step = 000493, loss = 0.001119
grad_step = 000494, loss = 0.001123
grad_step = 000495, loss = 0.001127
grad_step = 000496, loss = 0.001129
grad_step = 000497, loss = 0.001132
grad_step = 000498, loss = 0.001128
grad_step = 000499, loss = 0.001120
grad_step = 000500, loss = 0.001106
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001095
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
  date_run                              2020-05-09 10:51:37.925423
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.217119
metric_name                                  mean_absolute_error
Name: 0, dtype: object 
  date_run                              2020-05-09 10:51:37.931422
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.103445
metric_name                                   mean_squared_error
Name: 1, dtype: object 
  date_run                              2020-05-09 10:51:37.937743
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.139609
metric_name                                median_absolute_error
Name: 2, dtype: object 
  date_run                              2020-05-09 10:51:37.944002
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -0.57188
metric_name                                             r2_score
Name: 3, dtype: object 
  


### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10} 
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f47d423f3c8> <class 'mlmodels.model_keras.armdn.Model'>
  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355255.6875
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 276200.5938
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 204632.0781
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 142976.6719
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 98164.8203
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 68369.2188
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 48703.7734
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 35462.4219
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 26248.4785
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 19697.1738
  #### Inference Need return ypred, ytrue ######################### 
[[-2.6469412e-01 -8.5060251e-01 -2.6423731e-01  1.1739200e+00
   1.2470757e+00  9.1593796e-01 -9.5727283e-01 -2.3569357e-01
   6.3871855e-01  2.9869393e-01  1.1577857e+00 -6.9530618e-01
   7.3307753e-01 -5.5097795e-01  5.4545033e-01  1.1107235e+00
  -2.7471456e-01  8.2727748e-01 -8.7757069e-01  9.5414966e-02
  -1.2801166e+00  1.6090817e+00  2.3735182e-01  6.8769500e-02
   6.5648937e-01  2.5745395e-01 -4.6030009e-01 -4.9390322e-01
   1.3311448e+00 -1.0740182e+00 -8.0282897e-02  2.1810675e-01
   1.3640213e+00 -8.2822597e-01  1.4187671e+00  4.9489737e-01
  -2.3099676e-01  9.6933579e-01  7.4071980e-01 -5.4453325e-01
   2.4596605e-01 -8.4260714e-01  1.1192917e+00  8.9496769e-02
   4.9254525e-01  2.4539199e-01  9.5998442e-01 -3.8960156e-01
   1.0139209e+00 -1.0025325e+00 -3.8754597e-01  8.5351422e-02
   1.5605839e-01 -3.5212442e-02  8.8485771e-01 -1.0518157e+00
  -3.5241589e-02 -4.1680336e-02 -1.4146514e+00 -9.4412142e-01
  -2.9635921e-02  5.2366066e+00  4.1398668e+00  3.3077016e+00
   3.2843084e+00  4.2462263e+00  4.8080516e+00  4.2579794e+00
   4.6101131e+00  4.7578983e+00  4.6402149e+00  4.6348901e+00
   4.2043371e+00  4.1529398e+00  4.7710023e+00  3.5253377e+00
   3.4367890e+00  3.2727532e+00  3.0591359e+00  4.5462141e+00
   4.4005613e+00  3.9638443e+00  3.6460600e+00  2.9850645e+00
   2.9669571e+00  2.8180294e+00  4.9917655e+00  5.2151093e+00
   2.8987222e+00  2.9713786e+00  3.3650298e+00  5.1085815e+00
   4.2574668e+00  4.1389360e+00  3.5689437e+00  3.2100482e+00
   3.2162790e+00  3.3575571e+00  4.5312786e+00  3.1066196e+00
   3.8144684e+00  3.6578746e+00  5.4132996e+00  4.2293434e+00
   3.3512645e+00  4.1759377e+00  3.8964815e+00  3.8280337e+00
   5.0688114e+00  5.2081432e+00  3.9630115e+00  4.9583611e+00
   2.9034715e+00  3.4599485e+00  3.3486979e+00  2.7964544e+00
   3.6375713e+00  3.9065905e+00  3.8133955e+00  4.3815556e+00
   1.0403765e+00 -1.4753016e+00 -6.5441823e-01 -1.0619249e+00
  -1.0007638e+00  1.7868090e-01 -1.0661991e+00  1.1768371e+00
   1.1710234e+00 -1.9234097e-01  6.1793551e-03  5.3885299e-01
   1.1472245e+00  9.8289818e-01  1.0476393e+00 -6.9494218e-02
   1.1316454e+00 -2.9657197e-01 -6.5833032e-01 -1.8597847e-01
   3.9111301e-01  3.8038680e-01 -1.4871076e+00  2.6946616e-01
  -1.0383315e+00  5.3564554e-01  9.0428454e-01 -1.0194290e+00
  -1.0526856e+00  4.2276138e-01 -4.3161893e-01  1.2556407e+00
  -9.1097432e-01  1.2192501e+00  7.8061587e-01  1.3298274e+00
   1.2835443e+00  7.4905437e-01 -5.2427649e-01  6.0816419e-01
   1.1486976e+00 -2.8976566e-01 -7.9129958e-01 -1.7428687e-01
   3.8585934e-01 -3.0073699e-01  5.9755075e-01  5.9927154e-01
   1.9328515e-01 -5.1598984e-01 -9.4013119e-01  7.2309005e-01
  -3.4366804e-01 -9.0985316e-01  9.4953501e-01 -7.8515071e-01
   6.5605454e-02 -1.0606226e+00 -1.5332200e-01  2.8630435e-02
   7.3320323e-01  3.0105913e-01  1.9012980e+00  6.0022753e-01
   6.2710589e-01  2.0328374e+00  1.0309793e+00  2.5532405e+00
   3.2667327e-01  1.5722365e+00  2.0529652e+00  1.9212189e+00
   4.9344456e-01  2.9706371e-01  5.9086573e-01  2.7531654e-01
   2.1634140e+00  3.6620134e-01  1.0902057e+00  2.1431389e+00
   2.3514742e-01  1.6322178e+00  2.6161027e-01  1.5435221e+00
   4.4848764e-01  1.3500612e+00  1.1850619e+00  9.0761358e-01
   1.0925080e+00  2.6929128e-01  3.9677280e-01  1.2948452e+00
   1.9886506e-01  6.0164428e-01  1.7551923e+00  1.8850696e+00
   8.1153703e-01  1.9223194e+00  6.6454720e-01  6.0204303e-01
   4.2357314e-01  1.9377319e+00  4.3038952e-01  1.0917454e+00
   2.0137334e+00  3.1394011e-01  2.1628146e+00  1.4394366e+00
   1.9442356e-01  4.7907710e-01  1.4707315e+00  1.4920734e+00
   1.8104539e+00  6.5208524e-01  3.3663011e-01  1.4488199e+00
   8.3997512e-01  1.0573400e+00  1.5407407e+00  1.0791099e+00
   5.4734230e-02  5.6848464e+00  5.5178313e+00  5.8156519e+00
   5.1049585e+00  4.3987269e+00  5.0837450e+00  4.7302008e+00
   4.3694143e+00  5.3690667e+00  3.7259707e+00  3.9706650e+00
   5.2466202e+00  5.1060791e+00  4.4728351e+00  3.8917742e+00
   5.4205365e+00  4.2910938e+00  5.3524580e+00  4.7514896e+00
   5.7946749e+00  4.3146820e+00  4.9146032e+00  4.8200374e+00
   4.0611205e+00  5.7370958e+00  4.4870110e+00  5.2070465e+00
   4.8778930e+00  5.3321104e+00  5.7910419e+00  5.9242997e+00
   5.3128300e+00  4.4849596e+00  5.7691712e+00  3.8685489e+00
   5.2191405e+00  5.9720039e+00  5.2887173e+00  4.0941525e+00
   4.0225306e+00  4.4210033e+00  5.8341513e+00  6.0853539e+00
   4.7407603e+00  4.1071658e+00  6.2652555e+00  4.9311781e+00
   5.3996572e+00  6.0471287e+00  6.2783771e+00  4.9331875e+00
   6.1852708e+00  4.4566636e+00  4.7108960e+00  4.3184781e+00
   4.9235454e+00  5.7686343e+00  3.8156252e+00  4.1294904e+00
   1.9195032e-01  1.9216456e+00  6.2396854e-01  1.0532684e+00
   1.5572660e+00  5.2174550e-01  2.1458693e+00  6.3954633e-01
   7.2385013e-01  1.7436011e+00  2.2449894e+00  6.3611352e-01
   1.9257427e+00  1.3425819e+00  2.7240056e-01  4.8720908e-01
   1.7121155e+00  7.8709769e-01  2.8804696e-01  3.7411857e-01
   6.4366019e-01  2.0112085e-01  2.6269400e-01  1.6656923e+00
   1.3143742e+00  1.5883019e+00  2.2564802e+00  9.8988032e-01
   2.3531491e-01  4.5001829e-01  5.3183556e-01  1.9298875e+00
   7.7799582e-01  1.5582731e+00  9.5492184e-01  5.9707546e-01
   4.0344119e-01  8.5472965e-01  2.1176519e+00  7.9618895e-01
   1.2278142e+00  1.2004390e+00  1.9940946e+00  5.3623796e-01
   2.1617064e+00  6.5917981e-01  1.9389999e+00  8.5131061e-01
   1.9045019e+00  4.9612856e-01  5.3852272e-01  2.9751372e-01
   1.0066702e+00  6.7305803e-01  1.5248537e+00  1.1999153e+00
   1.0341725e+00  4.6697330e-01  3.7535131e-01  2.1674125e+00
  -1.4147494e+00  5.4912210e+00 -6.4253435e+00]]
  ### Calculate Metrics    ######################################## 
  date_run                              2020-05-09 10:51:46.170861
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   98.2047
metric_name                                  mean_absolute_error
Name: 4, dtype: object 
  date_run                              2020-05-09 10:51:46.175372
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9660.07
metric_name                                   mean_squared_error
Name: 5, dtype: object 
  date_run                              2020-05-09 10:51:46.179252
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   98.3847
metric_name                                median_absolute_error
Name: 6, dtype: object 
  date_run                              2020-05-09 10:51:46.182724
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -864.128
metric_name                                             r2_score
Name: 7, dtype: object 
  


### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ##### 
  #### Model URI and Config JSON 
  {'model_uri': 'model_gluon/fb_prophet.py'} 
  #### Setup Model   ############################################## 
  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f47cde00d68> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} [Errno 2] File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist: b'dataset/timeseries/stock/qqq_us_train.csv' 
  


### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ##### 
  #### Model URI and Config JSON 
  {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}} 
  #### Setup Model   ############################################## 
  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/ 
                       date_run  ...            metric_name
0  2020-05-09 10:51:37.925423  ...    mean_absolute_error
1  2020-05-09 10:51:37.931422  ...     mean_squared_error
2  2020-05-09 10:51:37.937743  ...  median_absolute_error
3  2020-05-09 10:51:37.944002  ...               r2_score
4  2020-05-09 10:51:46.170861  ...    mean_absolute_error
5  2020-05-09 10:51:46.175372  ...     mean_squared_error
6  2020-05-09 10:51:46.179252  ...  median_absolute_error
7  2020-05-09 10:51:46.182724  ...               r2_score

[8 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 124, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 89, in fit
    train_df, test_df = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 32, in get_dataset
    train_df = pd.read_csv(data_pars["train_data_path"], parse_dates=True)[col]
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
FileNotFoundError: [Errno 2] File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist: b'dataset/timeseries/stock/qqq_us_train.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 116, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

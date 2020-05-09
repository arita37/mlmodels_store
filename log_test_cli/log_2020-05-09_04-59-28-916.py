  ('/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json',) 
  ('test_cli', 'GITHUB_REPOSITORT', 'GITHUB_SHA') 
  ('Running command', 'test_cli') 
  ('# Testing Command Line System  ',) 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1234abd470b8c8275abee681fee18867d7ff18bb', 'url_branch_file': 'https://github.com/{repo}/blob/{branch}/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '1234abd470b8c8275abee681fee18867d7ff18bb', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1234abd470b8c8275abee681fee18867d7ff18bb

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1234abd470b8c8275abee681fee18867d7ff18bb

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
  (['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n', '- ml_optim     :  Hyper-parameter search\n', '- ml_test      :  Testing for developpers.\n'],) 





 ************************************************************************************************************************
ml_models --do init  --path ztest/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
init
  ('Working Folder', 'ztest/') 
  ('Config values', {'model_trained': 'ztest//model_trained/', 'dataset': 'ztest//dataset/'}) 
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
  ('ztest/',) 
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
  ('Fit',) 
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
  ('#### Module init   ############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
  (<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>,) 
  ('#### Loading params   ##############################################',) 
  ('############# Data, Params preparation   #################',) 
  ('#### Model init   ############################################',) 
  (<mlmodels.model_tf.1_lstm.Model object at 0x7f98f3b717f0>,) 
  ('#### Fit   ########################################################',) 
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
  ('#### Predict   ####################################################',) 
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
 [ 0.03441346 -0.01901797  0.10872503  0.03799187  0.10676638  0.0954449 ]
 [-0.09734005  0.00903079 -0.08819512  0.03629164 -0.07921486  0.16592188]
 [ 0.05143019  0.24882835  0.07539938 -0.06294081 -0.01797601  0.06363721]
 [ 0.2878089   0.28008151  0.30133364  0.21771342  0.14306197  0.1504769 ]
 [-0.09638313  0.50130475  0.20724994  0.20997055  0.52709281  0.04396925]
 [ 0.15008545  0.42706114 -0.05019376  0.31218335  0.25863266  0.09938888]
 [ 0.40838587  0.1807856  -0.25490221 -0.20669265 -0.06214004  0.38600796]
 [ 0.01097483 -0.26728457  0.46012121  0.49551558  0.18902794 -0.08149076]
 [ 0.          0.          0.          0.          0.          0.        ]]
  ('#### Get  metrics   ################################################',) 
  ('#### Save   ########################################################',) 
  ('#### Load   ########################################################',) 
model_tf.1_lstm
model_tf.1_lstm
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
  ('#### Loading params   ##############################################',) 
  ('############# Data, Params preparation   #################',) 
  ({'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6}, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}, {}, {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}) 
  ('#### Loading dataset   #############################################',) 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
  ('#### Model init  #############################################',) 
  ('#### Model fit   #############################################',) 
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
  ('#### Predict   #####################################################',) 
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
  ('#### metrics   #####################################################',) 
{'loss': 0.4612498302012682, 'loss_history': []}
  ('#### Plot   ########################################################',) 
  ('#### Save/Load   ###################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
model_tf.1_lstm
model_tf.1_lstm
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
  ('#### Loading params   ##############################################',) 
  ('############# Data, Params preparation   #################',) 
  ({'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6}, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}, {}, {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}) 
  ('#### Loading dataset   #############################################',) 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
  ('#### Model init  #############################################',) 
  ('#### Model fit   #############################################',) 
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
  ('#### Predict   #####################################################',) 
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
  ('#### metrics   #####################################################',) 
{'loss': 0.41608062759041786, 'loss_history': []}
  ('#### Plot   ########################################################',) 
  ('#### Save/Load   ###################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





 ************************************************************************************************************************
ml_models --do test  --model_uri "ztest/mycustom/my_lstm.py"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
test
  ('#### Module init   ############################################',) 
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
  ({'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}, {'engine': 'optuna', 'method': 'prune', 'ntrials': 5}, {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}}) 
  (<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>,) 
  ('###### Hyper-optimization through study   ##################################',) 
  ('check', <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}) 
[32m[I 2020-05-09 04:59:58,260][0m Finished trial#0 resulted in value: 8.61885118484497. Current best value is 8.61885118484497 with parameters: {'learning_rate': 0.046815555126297495, 'num_layers': 4, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
  ('check', <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}) 
[32m[I 2020-05-09 04:59:59,432][0m Finished trial#1 resulted in value: 0.5424976795911789. Current best value is 0.5424976795911789 with parameters: {'learning_rate': 0.019460089862658906, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
 ################################### ('Optim, finished',) ###################################
  ('### Save Stats   ##########################################################',) 
  ('### Run Model with best   #################################################',) 
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
  ('#### Saving     ###########################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/optim_1lstm/', 'model_type': 'model_tf', 'model_uri': 'model_tf-1_lstm'}





 ************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test02/model_list.json  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
  ('dataset/json/benchmark.json',) 
  ('Custom benchmark',) 
  (['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'],) 
  ('Model List', [{'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}}]) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon/fb_prophet.py'},) 
  ('#### Setup Model   ##############################################',) 
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fcb95775390> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  ({'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, FileNotFoundError(2, "File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist")) 
  ("### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10},) 
  ('#### Setup Model   ##############################################',) 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
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
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fcb8aa72128> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 359651.8125
Epoch 2/10

1/1 [==============================] - 0s 111ms/step - loss: 295946.8125
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 224441.4219
Epoch 4/10

1/1 [==============================] - 0s 111ms/step - loss: 149686.9375
Epoch 5/10

1/1 [==============================] - 0s 97ms/step - loss: 93129.6328
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 57694.8281
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 36704.2227
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 24123.6016
Epoch 9/10

1/1 [==============================] - 0s 108ms/step - loss: 16345.1982
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 11396.6494
  ('#### Inference Need return ypred, ytrue #########################',) 
[[ 1.00408122e-01 -2.31535986e-01  5.78083277e-01  3.39365005e-01
  -3.90237242e-01  1.10670485e-01 -7.42539048e-01  1.35839975e+00
   8.59586895e-02  4.62974370e-01  6.27917886e-01  1.26018691e+00
  -1.76538920e+00  2.64813483e-01 -4.58164036e-01  1.47743559e+00
   4.32245433e-01 -8.71669233e-01 -1.75673890e+00 -7.74558127e-01
   1.06713760e+00  1.64993450e-01  1.63123280e-01 -4.24382508e-01
   2.65823245e-01  5.59361041e-01 -1.02553236e+00  9.24091578e-01
  -1.17921174e+00 -3.43882263e-01 -1.34496796e+00 -7.30396390e-01
   1.25621486e+00 -5.44501781e-01 -2.02923402e-01 -8.16418946e-01
  -6.54816270e-01  1.15029424e-01  8.52046072e-01 -3.53714824e-01
   8.17617476e-01 -1.62285298e-01 -1.70204997e+00  1.92132279e-01
  -3.82691920e-01  4.26911831e-01  5.92705846e-01 -1.58167183e-02
   1.32544708e+00  1.84124246e-01 -5.52305132e-02  8.11110556e-01
  -9.93223965e-01 -2.90918857e-01  2.67345220e-01  1.12850738e+00
  -9.87683296e-01 -1.06994224e+00 -1.82250381e-01  1.52144149e-01
  -1.03182539e-01  4.19101715e+00  7.40886974e+00  6.99724722e+00
   6.10369825e+00  5.08439302e+00  4.96376944e+00  5.79051304e+00
   6.36753845e+00  6.17246294e+00  7.11020422e+00  6.52943373e+00
   6.31044149e+00  7.88464451e+00  7.62609291e+00  5.32597351e+00
   5.85622215e+00  6.35011864e+00  6.45948124e+00  6.17770243e+00
   6.97499752e+00  4.54067516e+00  6.40165472e+00  6.15693140e+00
   6.35857391e+00  6.02418709e+00  5.37820292e+00  5.02498913e+00
   5.79266357e+00  5.19591713e+00  5.43317366e+00  5.77237415e+00
   5.52331829e+00  5.74503326e+00  7.79628468e+00  5.78261137e+00
   5.98570395e+00  7.36647320e+00  6.91972876e+00  4.42453241e+00
   5.56375360e+00  6.96075916e+00  5.93240070e+00  5.76104164e+00
   5.16728830e+00  5.50933838e+00  6.26078129e+00  5.91935110e+00
   6.57440424e+00  5.24064159e+00  6.54814768e+00  5.84949780e+00
   6.28744888e+00  6.52740860e+00  5.42039871e+00  6.14431524e+00
   6.62188101e+00  4.87246990e+00  6.67133665e+00  5.92687225e+00
   7.59595692e-01  3.34254086e-01 -1.19578338e+00  3.36398125e-01
  -3.60530227e-01 -1.64325237e-01  2.73962975e-01 -2.56105959e-01
  -3.34980071e-01 -3.27562511e-01 -3.51442248e-02  1.74505532e-01
   3.02670777e-01  4.22244549e-01  1.49668217e+00 -8.18821311e-01
  -9.83152509e-01  7.73359418e-01 -1.03463292e+00  2.84892499e-01
  -9.35488164e-01 -1.32513344e-02  1.28690052e+00 -1.48093832e+00
   7.18540549e-01  5.99697232e-03 -5.90075135e-01 -7.38009214e-01
  -1.22541875e-01  1.26604795e-01 -5.15912116e-01 -8.31060261e-02
   5.83205462e-01  1.14168215e+00  1.59366965e-01 -7.93490410e-02
   1.39256060e+00 -2.02739805e-01 -1.54026616e+00 -9.81147662e-02
   1.09576333e+00 -8.89017522e-01  7.54552841e-01 -3.31370473e-01
   6.70412064e-01  2.56162345e-01  1.15676805e-01  9.36349392e-01
  -2.82225996e-01  1.31449842e+00 -7.59295702e-01 -4.49911654e-01
  -5.03164947e-01 -4.11031187e-01  6.31075442e-01  4.36618805e-01
   1.07646048e+00 -1.19088531e-01 -8.76044869e-01 -1.33314759e-01
   8.92587900e-01  3.38136137e-01  8.95889938e-01  1.76689470e+00
   7.84188032e-01  4.31631148e-01  7.27243483e-01  1.19949102e+00
   5.99237740e-01  2.88859034e+00  1.21999347e+00  1.22155344e+00
   6.57283068e-01  9.16000724e-01  2.11919284e+00  8.00665081e-01
   1.81616068e+00  2.17092896e+00  2.45407867e+00  1.47728896e+00
   1.92203832e+00  1.72147882e+00  6.06407642e-01  2.42106819e+00
   2.15315342e+00  2.07354116e+00  9.01163459e-01  7.87488818e-01
   4.93682861e-01  1.41854048e+00  4.31676269e-01  9.79676545e-01
   3.82793784e-01  7.92770028e-01  1.71693778e+00  6.35068893e-01
   1.62889409e+00  4.45337594e-01  9.58685994e-01  9.71943617e-01
   3.70938301e-01  8.21599364e-01  1.22196615e+00  5.93390584e-01
   4.03050840e-01  1.74898005e+00  8.82938921e-01  1.15317082e+00
   3.55865359e-01  1.27470112e+00  1.02858615e+00  3.74415576e-01
   6.70545638e-01  1.28561974e+00  6.15238667e-01  3.57339859e-01
   1.66641414e+00  5.00838041e-01  3.93486977e-01  2.15248537e+00
   3.70790958e-02  7.36205578e+00  6.84052944e+00  7.13644218e+00
   7.01363325e+00  6.49490356e+00  7.14685631e+00  6.76554775e+00
   6.51744461e+00  6.81517410e+00  6.35468769e+00  5.62206030e+00
   5.72956228e+00  6.13603210e+00  6.97921562e+00  6.26053095e+00
   6.91856050e+00  7.25696182e+00  5.26746607e+00  6.78026533e+00
   6.41756964e+00  7.55161285e+00  7.16595125e+00  5.62778711e+00
   7.26262999e+00  6.47406387e+00  5.93024588e+00  6.82639170e+00
   7.19114971e+00  6.00488663e+00  7.59059906e+00  6.55794573e+00
   5.68999338e+00  6.02546644e+00  5.68173599e+00  6.48382282e+00
   7.01894951e+00  7.74963474e+00  5.33185720e+00  6.82868195e+00
   6.11582994e+00  6.60350466e+00  7.61000776e+00  8.01019859e+00
   6.87912655e+00  7.78299618e+00  6.78900433e+00  7.34862804e+00
   6.46035957e+00  6.81774664e+00  6.81280661e+00  6.33653545e+00
   7.32609415e+00  6.96239758e+00  8.16384602e+00  6.23144627e+00
   6.73825788e+00  6.11112928e+00  6.58287525e+00  6.27674484e+00
   5.55634379e-01  2.49541092e+00  2.20615268e-01  6.20159686e-01
   5.43236017e-01  1.30812001e+00  2.12699711e-01  7.34929383e-01
   1.04178834e+00  3.98579121e-01  1.89218032e+00  1.41620898e+00
   6.81503475e-01  4.43303108e-01  6.06638670e-01  9.59864199e-01
   8.40585470e-01  1.94145918e+00  5.93511283e-01  3.15192699e-01
   7.08074927e-01  2.21716499e+00  1.32816601e+00  3.27612281e-01
   2.62207651e+00  8.07364821e-01  2.46200633e+00  5.08914411e-01
   1.43227088e+00  6.69760346e-01  9.81065392e-01  1.38784337e+00
   1.43043375e+00  1.27574396e+00  5.69106817e-01  8.39956522e-01
   1.97593689e-01  1.74917388e+00  7.15648293e-01  1.31949067e+00
   1.65367699e+00  4.08325672e-01  2.32613206e-01  9.85944867e-01
   1.52699804e+00  1.46668935e+00  1.46960449e+00  3.74770701e-01
   1.83570671e+00  1.42075539e+00  1.11643755e+00  8.05819929e-01
   4.50627327e-01  6.19885504e-01  8.34138513e-01  2.01357889e+00
   6.47862136e-01  1.99507356e+00  1.48615837e+00  2.83031988e+00
  -9.89067268e+00  2.39609528e+00 -5.56721163e+00]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 05:00:12.244869
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.6002
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 05:00:12.249726
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9159.61
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 05:00:12.253569
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.5724
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 05:00:12.256602
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -819.308
metric_name                                             r2_score
Name: 3, dtype: object,) 
  ("### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256},) 
  ('#### Setup Model   ##############################################',) 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140511847273976
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140510840786112
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140510840786616
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140510840377584
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140510840378088
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140510840378592
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fcb95775400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.608845
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.569727
grad_step = 000002, loss = 0.540236
grad_step = 000003, loss = 0.509545
grad_step = 000004, loss = 0.477333
grad_step = 000005, loss = 0.446122
grad_step = 000006, loss = 0.431120
grad_step = 000007, loss = 0.428905
grad_step = 000008, loss = 0.415033
grad_step = 000009, loss = 0.395265
grad_step = 000010, loss = 0.380049
grad_step = 000011, loss = 0.370838
grad_step = 000012, loss = 0.364218
grad_step = 000013, loss = 0.356873
grad_step = 000014, loss = 0.347447
grad_step = 000015, loss = 0.336121
grad_step = 000016, loss = 0.324063
grad_step = 000017, loss = 0.313071
grad_step = 000018, loss = 0.304578
grad_step = 000019, loss = 0.297273
grad_step = 000020, loss = 0.288364
grad_step = 000021, loss = 0.278138
grad_step = 000022, loss = 0.268705
grad_step = 000023, loss = 0.260696
grad_step = 000024, loss = 0.253264
grad_step = 000025, loss = 0.245564
grad_step = 000026, loss = 0.237344
grad_step = 000027, loss = 0.228870
grad_step = 000028, loss = 0.220651
grad_step = 000029, loss = 0.213087
grad_step = 000030, loss = 0.206146
grad_step = 000031, loss = 0.199288
grad_step = 000032, loss = 0.192126
grad_step = 000033, loss = 0.184870
grad_step = 000034, loss = 0.177974
grad_step = 000035, loss = 0.171550
grad_step = 000036, loss = 0.165342
grad_step = 000037, loss = 0.159175
grad_step = 000038, loss = 0.153015
grad_step = 000039, loss = 0.146980
grad_step = 000040, loss = 0.141327
grad_step = 000041, loss = 0.136058
grad_step = 000042, loss = 0.130801
grad_step = 000043, loss = 0.125463
grad_step = 000044, loss = 0.120357
grad_step = 000045, loss = 0.115599
grad_step = 000046, loss = 0.111001
grad_step = 000047, loss = 0.106419
grad_step = 000048, loss = 0.101954
grad_step = 000049, loss = 0.097782
grad_step = 000050, loss = 0.093834
grad_step = 000051, loss = 0.089898
grad_step = 000052, loss = 0.086009
grad_step = 000053, loss = 0.082351
grad_step = 000054, loss = 0.078899
grad_step = 000055, loss = 0.075522
grad_step = 000056, loss = 0.072221
grad_step = 000057, loss = 0.069080
grad_step = 000058, loss = 0.066106
grad_step = 000059, loss = 0.063210
grad_step = 000060, loss = 0.060378
grad_step = 000061, loss = 0.057675
grad_step = 000062, loss = 0.055096
grad_step = 000063, loss = 0.052616
grad_step = 000064, loss = 0.050224
grad_step = 000065, loss = 0.047926
grad_step = 000066, loss = 0.045727
grad_step = 000067, loss = 0.043612
grad_step = 000068, loss = 0.041575
grad_step = 000069, loss = 0.039621
grad_step = 000070, loss = 0.037759
grad_step = 000071, loss = 0.035971
grad_step = 000072, loss = 0.034249
grad_step = 000073, loss = 0.032611
grad_step = 000074, loss = 0.031045
grad_step = 000075, loss = 0.029534
grad_step = 000076, loss = 0.028094
grad_step = 000077, loss = 0.026730
grad_step = 000078, loss = 0.025419
grad_step = 000079, loss = 0.024159
grad_step = 000080, loss = 0.022969
grad_step = 000081, loss = 0.021833
grad_step = 000082, loss = 0.020741
grad_step = 000083, loss = 0.019706
grad_step = 000084, loss = 0.018727
grad_step = 000085, loss = 0.017787
grad_step = 000086, loss = 0.016893
grad_step = 000087, loss = 0.016048
grad_step = 000088, loss = 0.015240
grad_step = 000089, loss = 0.014473
grad_step = 000090, loss = 0.013748
grad_step = 000091, loss = 0.013059
grad_step = 000092, loss = 0.012404
grad_step = 000093, loss = 0.011785
grad_step = 000094, loss = 0.011198
grad_step = 000095, loss = 0.010641
grad_step = 000096, loss = 0.010116
grad_step = 000097, loss = 0.009619
grad_step = 000098, loss = 0.009148
grad_step = 000099, loss = 0.008704
grad_step = 000100, loss = 0.008285
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.007889
grad_step = 000102, loss = 0.007516
grad_step = 000103, loss = 0.007165
grad_step = 000104, loss = 0.006833
grad_step = 000105, loss = 0.006522
grad_step = 000106, loss = 0.006229
grad_step = 000107, loss = 0.005953
grad_step = 000108, loss = 0.005694
grad_step = 000109, loss = 0.005452
grad_step = 000110, loss = 0.005223
grad_step = 000111, loss = 0.005010
grad_step = 000112, loss = 0.004810
grad_step = 000113, loss = 0.004622
grad_step = 000114, loss = 0.004446
grad_step = 000115, loss = 0.004282
grad_step = 000116, loss = 0.004128
grad_step = 000117, loss = 0.003985
grad_step = 000118, loss = 0.003851
grad_step = 000119, loss = 0.003726
grad_step = 000120, loss = 0.003610
grad_step = 000121, loss = 0.003501
grad_step = 000122, loss = 0.003400
grad_step = 000123, loss = 0.003306
grad_step = 000124, loss = 0.003218
grad_step = 000125, loss = 0.003137
grad_step = 000126, loss = 0.003061
grad_step = 000127, loss = 0.002990
grad_step = 000128, loss = 0.002924
grad_step = 000129, loss = 0.002863
grad_step = 000130, loss = 0.002807
grad_step = 000131, loss = 0.002754
grad_step = 000132, loss = 0.002705
grad_step = 000133, loss = 0.002659
grad_step = 000134, loss = 0.002617
grad_step = 000135, loss = 0.002577
grad_step = 000136, loss = 0.002541
grad_step = 000137, loss = 0.002506
grad_step = 000138, loss = 0.002474
grad_step = 000139, loss = 0.002445
grad_step = 000140, loss = 0.002417
grad_step = 000141, loss = 0.002392
grad_step = 000142, loss = 0.002368
grad_step = 000143, loss = 0.002345
grad_step = 000144, loss = 0.002324
grad_step = 000145, loss = 0.002304
grad_step = 000146, loss = 0.002286
grad_step = 000147, loss = 0.002269
grad_step = 000148, loss = 0.002253
grad_step = 000149, loss = 0.002239
grad_step = 000150, loss = 0.002225
grad_step = 000151, loss = 0.002212
grad_step = 000152, loss = 0.002199
grad_step = 000153, loss = 0.002187
grad_step = 000154, loss = 0.002176
grad_step = 000155, loss = 0.002166
grad_step = 000156, loss = 0.002156
grad_step = 000157, loss = 0.002147
grad_step = 000158, loss = 0.002139
grad_step = 000159, loss = 0.002132
grad_step = 000160, loss = 0.002127
grad_step = 000161, loss = 0.002128
grad_step = 000162, loss = 0.002133
grad_step = 000163, loss = 0.002145
grad_step = 000164, loss = 0.002145
grad_step = 000165, loss = 0.002128
grad_step = 000166, loss = 0.002096
grad_step = 000167, loss = 0.002078
grad_step = 000168, loss = 0.002083
grad_step = 000169, loss = 0.002096
grad_step = 000170, loss = 0.002104
grad_step = 000171, loss = 0.002090
grad_step = 000172, loss = 0.002069
grad_step = 000173, loss = 0.002053
grad_step = 000174, loss = 0.002053
grad_step = 000175, loss = 0.002061
grad_step = 000176, loss = 0.002065
grad_step = 000177, loss = 0.002061
grad_step = 000178, loss = 0.002047
grad_step = 000179, loss = 0.002035
grad_step = 000180, loss = 0.002029
grad_step = 000181, loss = 0.002030
grad_step = 000182, loss = 0.002034
grad_step = 000183, loss = 0.002037
grad_step = 000184, loss = 0.002037
grad_step = 000185, loss = 0.002031
grad_step = 000186, loss = 0.002024
grad_step = 000187, loss = 0.002016
grad_step = 000188, loss = 0.002010
grad_step = 000189, loss = 0.002007
grad_step = 000190, loss = 0.002006
grad_step = 000191, loss = 0.002007
grad_step = 000192, loss = 0.002009
grad_step = 000193, loss = 0.002012
grad_step = 000194, loss = 0.002015
grad_step = 000195, loss = 0.002020
grad_step = 000196, loss = 0.002023
grad_step = 000197, loss = 0.002024
grad_step = 000198, loss = 0.002018
grad_step = 000199, loss = 0.002009
grad_step = 000200, loss = 0.001997
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001987
grad_step = 000202, loss = 0.001982
grad_step = 000203, loss = 0.001982
grad_step = 000204, loss = 0.001985
grad_step = 000205, loss = 0.001989
grad_step = 000206, loss = 0.001995
grad_step = 000207, loss = 0.002001
grad_step = 000208, loss = 0.002010
grad_step = 000209, loss = 0.002014
grad_step = 000210, loss = 0.002014
grad_step = 000211, loss = 0.002002
grad_step = 000212, loss = 0.001985
grad_step = 000213, loss = 0.001971
grad_step = 000214, loss = 0.001965
grad_step = 000215, loss = 0.001967
grad_step = 000216, loss = 0.001975
grad_step = 000217, loss = 0.001982
grad_step = 000218, loss = 0.001985
grad_step = 000219, loss = 0.001984
grad_step = 000220, loss = 0.001977
grad_step = 000221, loss = 0.001967
grad_step = 000222, loss = 0.001958
grad_step = 000223, loss = 0.001953
grad_step = 000224, loss = 0.001953
grad_step = 000225, loss = 0.001955
grad_step = 000226, loss = 0.001958
grad_step = 000227, loss = 0.001962
grad_step = 000228, loss = 0.001965
grad_step = 000229, loss = 0.001965
grad_step = 000230, loss = 0.001964
grad_step = 000231, loss = 0.001960
grad_step = 000232, loss = 0.001955
grad_step = 000233, loss = 0.001949
grad_step = 000234, loss = 0.001943
grad_step = 000235, loss = 0.001939
grad_step = 000236, loss = 0.001937
grad_step = 000237, loss = 0.001936
grad_step = 000238, loss = 0.001937
grad_step = 000239, loss = 0.001938
grad_step = 000240, loss = 0.001941
grad_step = 000241, loss = 0.001945
grad_step = 000242, loss = 0.001950
grad_step = 000243, loss = 0.001958
grad_step = 000244, loss = 0.001964
grad_step = 000245, loss = 0.001971
grad_step = 000246, loss = 0.001970
grad_step = 000247, loss = 0.001966
grad_step = 000248, loss = 0.001952
grad_step = 000249, loss = 0.001938
grad_step = 000250, loss = 0.001925
grad_step = 000251, loss = 0.001919
grad_step = 000252, loss = 0.001919
grad_step = 000253, loss = 0.001924
grad_step = 000254, loss = 0.001931
grad_step = 000255, loss = 0.001938
grad_step = 000256, loss = 0.001946
grad_step = 000257, loss = 0.001948
grad_step = 000258, loss = 0.001946
grad_step = 000259, loss = 0.001936
grad_step = 000260, loss = 0.001925
grad_step = 000261, loss = 0.001914
grad_step = 000262, loss = 0.001907
grad_step = 000263, loss = 0.001905
grad_step = 000264, loss = 0.001908
grad_step = 000265, loss = 0.001912
grad_step = 000266, loss = 0.001916
grad_step = 000267, loss = 0.001919
grad_step = 000268, loss = 0.001920
grad_step = 000269, loss = 0.001921
grad_step = 000270, loss = 0.001917
grad_step = 000271, loss = 0.001912
grad_step = 000272, loss = 0.001905
grad_step = 000273, loss = 0.001899
grad_step = 000274, loss = 0.001893
grad_step = 000275, loss = 0.001890
grad_step = 000276, loss = 0.001888
grad_step = 000277, loss = 0.001888
grad_step = 000278, loss = 0.001889
grad_step = 000279, loss = 0.001891
grad_step = 000280, loss = 0.001893
grad_step = 000281, loss = 0.001896
grad_step = 000282, loss = 0.001901
grad_step = 000283, loss = 0.001905
grad_step = 000284, loss = 0.001913
grad_step = 000285, loss = 0.001918
grad_step = 000286, loss = 0.001924
grad_step = 000287, loss = 0.001921
grad_step = 000288, loss = 0.001918
grad_step = 000289, loss = 0.001905
grad_step = 000290, loss = 0.001892
grad_step = 000291, loss = 0.001879
grad_step = 000292, loss = 0.001871
grad_step = 000293, loss = 0.001867
grad_step = 000294, loss = 0.001868
grad_step = 000295, loss = 0.001873
grad_step = 000296, loss = 0.001878
grad_step = 000297, loss = 0.001886
grad_step = 000298, loss = 0.001893
grad_step = 000299, loss = 0.001900
grad_step = 000300, loss = 0.001900
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001897
grad_step = 000302, loss = 0.001888
grad_step = 000303, loss = 0.001877
grad_step = 000304, loss = 0.001865
grad_step = 000305, loss = 0.001856
grad_step = 000306, loss = 0.001851
grad_step = 000307, loss = 0.001851
grad_step = 000308, loss = 0.001853
grad_step = 000309, loss = 0.001856
grad_step = 000310, loss = 0.001862
grad_step = 000311, loss = 0.001868
grad_step = 000312, loss = 0.001877
grad_step = 000313, loss = 0.001883
grad_step = 000314, loss = 0.001891
grad_step = 000315, loss = 0.001891
grad_step = 000316, loss = 0.001886
grad_step = 000317, loss = 0.001873
grad_step = 000318, loss = 0.001859
grad_step = 000319, loss = 0.001845
grad_step = 000320, loss = 0.001836
grad_step = 000321, loss = 0.001833
grad_step = 000322, loss = 0.001834
grad_step = 000323, loss = 0.001839
grad_step = 000324, loss = 0.001846
grad_step = 000325, loss = 0.001854
grad_step = 000326, loss = 0.001863
grad_step = 000327, loss = 0.001874
grad_step = 000328, loss = 0.001878
grad_step = 000329, loss = 0.001879
grad_step = 000330, loss = 0.001870
grad_step = 000331, loss = 0.001856
grad_step = 000332, loss = 0.001839
grad_step = 000333, loss = 0.001826
grad_step = 000334, loss = 0.001818
grad_step = 000335, loss = 0.001816
grad_step = 000336, loss = 0.001819
grad_step = 000337, loss = 0.001826
grad_step = 000338, loss = 0.001833
grad_step = 000339, loss = 0.001840
grad_step = 000340, loss = 0.001849
grad_step = 000341, loss = 0.001855
grad_step = 000342, loss = 0.001859
grad_step = 000343, loss = 0.001855
grad_step = 000344, loss = 0.001846
grad_step = 000345, loss = 0.001832
grad_step = 000346, loss = 0.001818
grad_step = 000347, loss = 0.001806
grad_step = 000348, loss = 0.001800
grad_step = 000349, loss = 0.001799
grad_step = 000350, loss = 0.001801
grad_step = 000351, loss = 0.001806
grad_step = 000352, loss = 0.001812
grad_step = 000353, loss = 0.001819
grad_step = 000354, loss = 0.001825
grad_step = 000355, loss = 0.001833
grad_step = 000356, loss = 0.001838
grad_step = 000357, loss = 0.001842
grad_step = 000358, loss = 0.001839
grad_step = 000359, loss = 0.001831
grad_step = 000360, loss = 0.001818
grad_step = 000361, loss = 0.001803
grad_step = 000362, loss = 0.001790
grad_step = 000363, loss = 0.001781
grad_step = 000364, loss = 0.001778
grad_step = 000365, loss = 0.001780
grad_step = 000366, loss = 0.001788
grad_step = 000367, loss = 0.001804
grad_step = 000368, loss = 0.001831
grad_step = 000369, loss = 0.001864
grad_step = 000370, loss = 0.001888
grad_step = 000371, loss = 0.001897
grad_step = 000372, loss = 0.001890
grad_step = 000373, loss = 0.001847
grad_step = 000374, loss = 0.001801
grad_step = 000375, loss = 0.001768
grad_step = 000376, loss = 0.001768
grad_step = 000377, loss = 0.001790
grad_step = 000378, loss = 0.001808
grad_step = 000379, loss = 0.001811
grad_step = 000380, loss = 0.001797
grad_step = 000381, loss = 0.001780
grad_step = 000382, loss = 0.001770
grad_step = 000383, loss = 0.001761
grad_step = 000384, loss = 0.001752
grad_step = 000385, loss = 0.001751
grad_step = 000386, loss = 0.001757
grad_step = 000387, loss = 0.001762
grad_step = 000388, loss = 0.001762
grad_step = 000389, loss = 0.001758
grad_step = 000390, loss = 0.001748
grad_step = 000391, loss = 0.001739
grad_step = 000392, loss = 0.001731
grad_step = 000393, loss = 0.001728
grad_step = 000394, loss = 0.001729
grad_step = 000395, loss = 0.001729
grad_step = 000396, loss = 0.001728
grad_step = 000397, loss = 0.001726
grad_step = 000398, loss = 0.001725
grad_step = 000399, loss = 0.001723
grad_step = 000400, loss = 0.001720
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001716
grad_step = 000402, loss = 0.001714
grad_step = 000403, loss = 0.001712
grad_step = 000404, loss = 0.001714
grad_step = 000405, loss = 0.001722
grad_step = 000406, loss = 0.001746
grad_step = 000407, loss = 0.001766
grad_step = 000408, loss = 0.001784
grad_step = 000409, loss = 0.001747
grad_step = 000410, loss = 0.001699
grad_step = 000411, loss = 0.001672
grad_step = 000412, loss = 0.001687
grad_step = 000413, loss = 0.001711
grad_step = 000414, loss = 0.001699
grad_step = 000415, loss = 0.001668
grad_step = 000416, loss = 0.001648
grad_step = 000417, loss = 0.001657
grad_step = 000418, loss = 0.001673
grad_step = 000419, loss = 0.001668
grad_step = 000420, loss = 0.001649
grad_step = 000421, loss = 0.001631
grad_step = 000422, loss = 0.001625
grad_step = 000423, loss = 0.001630
grad_step = 000424, loss = 0.001636
grad_step = 000425, loss = 0.001636
grad_step = 000426, loss = 0.001629
grad_step = 000427, loss = 0.001622
grad_step = 000428, loss = 0.001619
grad_step = 000429, loss = 0.001624
grad_step = 000430, loss = 0.001634
grad_step = 000431, loss = 0.001654
grad_step = 000432, loss = 0.001674
grad_step = 000433, loss = 0.001711
grad_step = 000434, loss = 0.001736
grad_step = 000435, loss = 0.001777
grad_step = 000436, loss = 0.001743
grad_step = 000437, loss = 0.001693
grad_step = 000438, loss = 0.001608
grad_step = 000439, loss = 0.001584
grad_step = 000440, loss = 0.001620
grad_step = 000441, loss = 0.001653
grad_step = 000442, loss = 0.001651
grad_step = 000443, loss = 0.001595
grad_step = 000444, loss = 0.001565
grad_step = 000445, loss = 0.001580
grad_step = 000446, loss = 0.001610
grad_step = 000447, loss = 0.001623
grad_step = 000448, loss = 0.001594
grad_step = 000449, loss = 0.001565
grad_step = 000450, loss = 0.001559
grad_step = 000451, loss = 0.001572
grad_step = 000452, loss = 0.001586
grad_step = 000453, loss = 0.001580
grad_step = 000454, loss = 0.001567
grad_step = 000455, loss = 0.001557
grad_step = 000456, loss = 0.001557
grad_step = 000457, loss = 0.001561
grad_step = 000458, loss = 0.001559
grad_step = 000459, loss = 0.001554
grad_step = 000460, loss = 0.001551
grad_step = 000461, loss = 0.001554
grad_step = 000462, loss = 0.001556
grad_step = 000463, loss = 0.001554
grad_step = 000464, loss = 0.001547
grad_step = 000465, loss = 0.001540
grad_step = 000466, loss = 0.001537
grad_step = 000467, loss = 0.001539
grad_step = 000468, loss = 0.001541
grad_step = 000469, loss = 0.001542
grad_step = 000470, loss = 0.001540
grad_step = 000471, loss = 0.001539
grad_step = 000472, loss = 0.001541
grad_step = 000473, loss = 0.001546
grad_step = 000474, loss = 0.001553
grad_step = 000475, loss = 0.001558
grad_step = 000476, loss = 0.001564
grad_step = 000477, loss = 0.001564
grad_step = 000478, loss = 0.001563
grad_step = 000479, loss = 0.001555
grad_step = 000480, loss = 0.001546
grad_step = 000481, loss = 0.001535
grad_step = 000482, loss = 0.001526
grad_step = 000483, loss = 0.001523
grad_step = 000484, loss = 0.001524
grad_step = 000485, loss = 0.001528
grad_step = 000486, loss = 0.001532
grad_step = 000487, loss = 0.001534
grad_step = 000488, loss = 0.001532
grad_step = 000489, loss = 0.001528
grad_step = 000490, loss = 0.001522
grad_step = 000491, loss = 0.001516
grad_step = 000492, loss = 0.001513
grad_step = 000493, loss = 0.001511
grad_step = 000494, loss = 0.001511
grad_step = 000495, loss = 0.001512
grad_step = 000496, loss = 0.001512
grad_step = 000497, loss = 0.001512
grad_step = 000498, loss = 0.001513
grad_step = 000499, loss = 0.001514
grad_step = 000500, loss = 0.001517
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001522
Finished.
  ('#### Inference Need return ypred, ytrue #########################',) 
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 05:00:30.918620
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.243426
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 05:00:30.923713
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.132058
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 05:00:30.930505
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.147165
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 05:00:30.935322
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.00667
metric_name                                             r2_score
Name: 7, dtype: object,) 
  ("### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}},) 
  ('#### Setup Model   ##############################################',) 
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  ({'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}},) 
  ('#### Setup Model   ##############################################',) 

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  ({'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/',) 
  (                     date_run  ...            metric_name
0  2020-05-09 05:00:12.244869  ...    mean_absolute_error
1  2020-05-09 05:00:12.249726  ...     mean_squared_error
2  2020-05-09 05:00:12.253569  ...  median_absolute_error
3  2020-05-09 05:00:12.256602  ...               r2_score
4  2020-05-09 05:00:30.918620  ...    mean_absolute_error
5  2020-05-09 05:00:30.923713  ...     mean_squared_error
6  2020-05-09 05:00:30.930505  ...  median_absolute_error
7  2020-05-09 05:00:30.935322  ...               r2_score

[8 rows x 6 columns],) 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range





 ************************************************************************************************************************
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
  ('dataset/json/benchmark.json',) 
  ('Custom benchmark',) 
  (['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'],) 
  ('Model List', [{'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}]) 
  ("### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256},) 
  ('#### Setup Model   ##############################################',) 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140134324995968
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140133046214552
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140133045760464
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140133045760968
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140133045761472
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140133045761976
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7390a9e630> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.473632
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.435298
grad_step = 000002, loss = 0.405895
grad_step = 000003, loss = 0.377819
grad_step = 000004, loss = 0.351546
grad_step = 000005, loss = 0.329078
grad_step = 000006, loss = 0.305217
grad_step = 000007, loss = 0.284839
grad_step = 000008, loss = 0.273386
grad_step = 000009, loss = 0.260614
grad_step = 000010, loss = 0.245879
grad_step = 000011, loss = 0.234047
grad_step = 000012, loss = 0.223190
grad_step = 000013, loss = 0.214258
grad_step = 000014, loss = 0.206484
grad_step = 000015, loss = 0.197023
grad_step = 000016, loss = 0.187251
grad_step = 000017, loss = 0.179263
grad_step = 000018, loss = 0.171894
grad_step = 000019, loss = 0.163210
grad_step = 000020, loss = 0.153137
grad_step = 000021, loss = 0.143899
grad_step = 000022, loss = 0.135668
grad_step = 000023, loss = 0.127310
grad_step = 000024, loss = 0.118680
grad_step = 000025, loss = 0.110862
grad_step = 000026, loss = 0.104679
grad_step = 000027, loss = 0.099564
grad_step = 000028, loss = 0.093990
grad_step = 000029, loss = 0.087990
grad_step = 000030, loss = 0.082622
grad_step = 000031, loss = 0.077808
grad_step = 000032, loss = 0.072827
grad_step = 000033, loss = 0.067599
grad_step = 000034, loss = 0.062646
grad_step = 000035, loss = 0.058232
grad_step = 000036, loss = 0.054052
grad_step = 000037, loss = 0.050001
grad_step = 000038, loss = 0.046315
grad_step = 000039, loss = 0.042941
grad_step = 000040, loss = 0.039692
grad_step = 000041, loss = 0.036519
grad_step = 000042, loss = 0.033471
grad_step = 000043, loss = 0.030615
grad_step = 000044, loss = 0.028046
grad_step = 000045, loss = 0.025756
grad_step = 000046, loss = 0.023580
grad_step = 000047, loss = 0.021507
grad_step = 000048, loss = 0.019640
grad_step = 000049, loss = 0.017930
grad_step = 000050, loss = 0.016312
grad_step = 000051, loss = 0.014832
grad_step = 000052, loss = 0.013487
grad_step = 000053, loss = 0.012275
grad_step = 000054, loss = 0.011200
grad_step = 000055, loss = 0.010185
grad_step = 000056, loss = 0.009237
grad_step = 000057, loss = 0.008428
grad_step = 000058, loss = 0.007694
grad_step = 000059, loss = 0.006997
grad_step = 000060, loss = 0.006404
grad_step = 000061, loss = 0.005887
grad_step = 000062, loss = 0.005402
grad_step = 000063, loss = 0.004975
grad_step = 000064, loss = 0.004594
grad_step = 000065, loss = 0.004239
grad_step = 000066, loss = 0.003925
grad_step = 000067, loss = 0.003680
grad_step = 000068, loss = 0.003446
grad_step = 000069, loss = 0.003261
grad_step = 000070, loss = 0.003090
grad_step = 000071, loss = 0.002936
grad_step = 000072, loss = 0.002806
grad_step = 000073, loss = 0.002694
grad_step = 000074, loss = 0.002599
grad_step = 000075, loss = 0.002531
grad_step = 000076, loss = 0.002483
grad_step = 000077, loss = 0.002439
grad_step = 000078, loss = 0.002399
grad_step = 000079, loss = 0.002368
grad_step = 000080, loss = 0.002341
grad_step = 000081, loss = 0.002321
grad_step = 000082, loss = 0.002310
grad_step = 000083, loss = 0.002303
grad_step = 000084, loss = 0.002298
grad_step = 000085, loss = 0.002290
grad_step = 000086, loss = 0.002286
grad_step = 000087, loss = 0.002281
grad_step = 000088, loss = 0.002275
grad_step = 000089, loss = 0.002273
grad_step = 000090, loss = 0.002271
grad_step = 000091, loss = 0.002266
grad_step = 000092, loss = 0.002260
grad_step = 000093, loss = 0.002253
grad_step = 000094, loss = 0.002246
grad_step = 000095, loss = 0.002238
grad_step = 000096, loss = 0.002230
grad_step = 000097, loss = 0.002222
grad_step = 000098, loss = 0.002212
grad_step = 000099, loss = 0.002201
grad_step = 000100, loss = 0.002190
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002180
grad_step = 000102, loss = 0.002170
grad_step = 000103, loss = 0.002160
grad_step = 000104, loss = 0.002150
grad_step = 000105, loss = 0.002139
grad_step = 000106, loss = 0.002129
grad_step = 000107, loss = 0.002120
grad_step = 000108, loss = 0.002110
grad_step = 000109, loss = 0.002102
grad_step = 000110, loss = 0.002093
grad_step = 000111, loss = 0.002085
grad_step = 000112, loss = 0.002077
grad_step = 000113, loss = 0.002069
grad_step = 000114, loss = 0.002061
grad_step = 000115, loss = 0.002054
grad_step = 000116, loss = 0.002048
grad_step = 000117, loss = 0.002041
grad_step = 000118, loss = 0.002035
grad_step = 000119, loss = 0.002030
grad_step = 000120, loss = 0.002024
grad_step = 000121, loss = 0.002017
grad_step = 000122, loss = 0.002011
grad_step = 000123, loss = 0.002005
grad_step = 000124, loss = 0.002000
grad_step = 000125, loss = 0.001995
grad_step = 000126, loss = 0.001992
grad_step = 000127, loss = 0.001989
grad_step = 000128, loss = 0.001986
grad_step = 000129, loss = 0.001980
grad_step = 000130, loss = 0.001972
grad_step = 000131, loss = 0.001965
grad_step = 000132, loss = 0.001962
grad_step = 000133, loss = 0.001961
grad_step = 000134, loss = 0.001958
grad_step = 000135, loss = 0.001954
grad_step = 000136, loss = 0.001947
grad_step = 000137, loss = 0.001940
grad_step = 000138, loss = 0.001936
grad_step = 000139, loss = 0.001934
grad_step = 000140, loss = 0.001933
grad_step = 000141, loss = 0.001931
grad_step = 000142, loss = 0.001930
grad_step = 000143, loss = 0.001926
grad_step = 000144, loss = 0.001920
grad_step = 000145, loss = 0.001912
grad_step = 000146, loss = 0.001906
grad_step = 000147, loss = 0.001902
grad_step = 000148, loss = 0.001900
grad_step = 000149, loss = 0.001899
grad_step = 000150, loss = 0.001899
grad_step = 000151, loss = 0.001900
grad_step = 000152, loss = 0.001902
grad_step = 000153, loss = 0.001901
grad_step = 000154, loss = 0.001895
grad_step = 000155, loss = 0.001883
grad_step = 000156, loss = 0.001872
grad_step = 000157, loss = 0.001867
grad_step = 000158, loss = 0.001867
grad_step = 000159, loss = 0.001869
grad_step = 000160, loss = 0.001869
grad_step = 000161, loss = 0.001866
grad_step = 000162, loss = 0.001858
grad_step = 000163, loss = 0.001849
grad_step = 000164, loss = 0.001842
grad_step = 000165, loss = 0.001838
grad_step = 000166, loss = 0.001836
grad_step = 000167, loss = 0.001835
grad_step = 000168, loss = 0.001834
grad_step = 000169, loss = 0.001832
grad_step = 000170, loss = 0.001828
grad_step = 000171, loss = 0.001822
grad_step = 000172, loss = 0.001816
grad_step = 000173, loss = 0.001808
grad_step = 000174, loss = 0.001802
grad_step = 000175, loss = 0.001796
grad_step = 000176, loss = 0.001792
grad_step = 000177, loss = 0.001788
grad_step = 000178, loss = 0.001784
grad_step = 000179, loss = 0.001781
grad_step = 000180, loss = 0.001781
grad_step = 000181, loss = 0.001786
grad_step = 000182, loss = 0.001796
grad_step = 000183, loss = 0.001816
grad_step = 000184, loss = 0.001827
grad_step = 000185, loss = 0.001822
grad_step = 000186, loss = 0.001780
grad_step = 000187, loss = 0.001743
grad_step = 000188, loss = 0.001738
grad_step = 000189, loss = 0.001755
grad_step = 000190, loss = 0.001766
grad_step = 000191, loss = 0.001746
grad_step = 000192, loss = 0.001718
grad_step = 000193, loss = 0.001709
grad_step = 000194, loss = 0.001718
grad_step = 000195, loss = 0.001723
grad_step = 000196, loss = 0.001709
grad_step = 000197, loss = 0.001691
grad_step = 000198, loss = 0.001687
grad_step = 000199, loss = 0.001699
grad_step = 000200, loss = 0.001709
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001690
grad_step = 000202, loss = 0.001674
grad_step = 000203, loss = 0.001687
grad_step = 000204, loss = 0.001690
grad_step = 000205, loss = 0.001668
grad_step = 000206, loss = 0.001663
grad_step = 000207, loss = 0.001684
grad_step = 000208, loss = 0.001699
grad_step = 000209, loss = 0.001713
grad_step = 000210, loss = 0.001755
grad_step = 000211, loss = 0.001842
grad_step = 000212, loss = 0.001825
grad_step = 000213, loss = 0.001765
grad_step = 000214, loss = 0.001664
grad_step = 000215, loss = 0.001645
grad_step = 000216, loss = 0.001703
grad_step = 000217, loss = 0.001723
grad_step = 000218, loss = 0.001675
grad_step = 000219, loss = 0.001625
grad_step = 000220, loss = 0.001648
grad_step = 000221, loss = 0.001683
grad_step = 000222, loss = 0.001656
grad_step = 000223, loss = 0.001621
grad_step = 000224, loss = 0.001622
grad_step = 000225, loss = 0.001638
grad_step = 000226, loss = 0.001644
grad_step = 000227, loss = 0.001617
grad_step = 000228, loss = 0.001597
grad_step = 000229, loss = 0.001610
grad_step = 000230, loss = 0.001620
grad_step = 000231, loss = 0.001609
grad_step = 000232, loss = 0.001593
grad_step = 000233, loss = 0.001585
grad_step = 000234, loss = 0.001585
grad_step = 000235, loss = 0.001591
grad_step = 000236, loss = 0.001592
grad_step = 000237, loss = 0.001581
grad_step = 000238, loss = 0.001568
grad_step = 000239, loss = 0.001562
grad_step = 000240, loss = 0.001559
grad_step = 000241, loss = 0.001559
grad_step = 000242, loss = 0.001560
grad_step = 000243, loss = 0.001564
grad_step = 000244, loss = 0.001568
grad_step = 000245, loss = 0.001574
grad_step = 000246, loss = 0.001580
grad_step = 000247, loss = 0.001597
grad_step = 000248, loss = 0.001611
grad_step = 000249, loss = 0.001635
grad_step = 000250, loss = 0.001638
grad_step = 000251, loss = 0.001636
grad_step = 000252, loss = 0.001591
grad_step = 000253, loss = 0.001544
grad_step = 000254, loss = 0.001505
grad_step = 000255, loss = 0.001496
grad_step = 000256, loss = 0.001513
grad_step = 000257, loss = 0.001539
grad_step = 000258, loss = 0.001566
grad_step = 000259, loss = 0.001575
grad_step = 000260, loss = 0.001571
grad_step = 000261, loss = 0.001535
grad_step = 000262, loss = 0.001499
grad_step = 000263, loss = 0.001469
grad_step = 000264, loss = 0.001457
grad_step = 000265, loss = 0.001460
grad_step = 000266, loss = 0.001474
grad_step = 000267, loss = 0.001495
grad_step = 000268, loss = 0.001518
grad_step = 000269, loss = 0.001554
grad_step = 000270, loss = 0.001577
grad_step = 000271, loss = 0.001603
grad_step = 000272, loss = 0.001580
grad_step = 000273, loss = 0.001539
grad_step = 000274, loss = 0.001470
grad_step = 000275, loss = 0.001424
grad_step = 000276, loss = 0.001417
grad_step = 000277, loss = 0.001440
grad_step = 000278, loss = 0.001477
grad_step = 000279, loss = 0.001499
grad_step = 000280, loss = 0.001506
grad_step = 000281, loss = 0.001476
grad_step = 000282, loss = 0.001441
grad_step = 000283, loss = 0.001408
grad_step = 000284, loss = 0.001392
grad_step = 000285, loss = 0.001394
grad_step = 000286, loss = 0.001407
grad_step = 000287, loss = 0.001426
grad_step = 000288, loss = 0.001446
grad_step = 000289, loss = 0.001472
grad_step = 000290, loss = 0.001485
grad_step = 000291, loss = 0.001500
grad_step = 000292, loss = 0.001480
grad_step = 000293, loss = 0.001457
grad_step = 000294, loss = 0.001414
grad_step = 000295, loss = 0.001383
grad_step = 000296, loss = 0.001369
grad_step = 000297, loss = 0.001375
grad_step = 000298, loss = 0.001392
grad_step = 000299, loss = 0.001409
grad_step = 000300, loss = 0.001422
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001416
grad_step = 000302, loss = 0.001405
grad_step = 000303, loss = 0.001385
grad_step = 000304, loss = 0.001369
grad_step = 000305, loss = 0.001357
grad_step = 000306, loss = 0.001354
grad_step = 000307, loss = 0.001357
grad_step = 000308, loss = 0.001363
grad_step = 000309, loss = 0.001372
grad_step = 000310, loss = 0.001382
grad_step = 000311, loss = 0.001396
grad_step = 000312, loss = 0.001407
grad_step = 000313, loss = 0.001427
grad_step = 000314, loss = 0.001433
grad_step = 000315, loss = 0.001444
grad_step = 000316, loss = 0.001427
grad_step = 000317, loss = 0.001408
grad_step = 000318, loss = 0.001373
grad_step = 000319, loss = 0.001346
grad_step = 000320, loss = 0.001336
grad_step = 000321, loss = 0.001345
grad_step = 000322, loss = 0.001362
grad_step = 000323, loss = 0.001375
grad_step = 000324, loss = 0.001388
grad_step = 000325, loss = 0.001390
grad_step = 000326, loss = 0.001393
grad_step = 000327, loss = 0.001378
grad_step = 000328, loss = 0.001360
grad_step = 000329, loss = 0.001339
grad_step = 000330, loss = 0.001328
grad_step = 000331, loss = 0.001325
grad_step = 000332, loss = 0.001328
grad_step = 000333, loss = 0.001332
grad_step = 000334, loss = 0.001336
grad_step = 000335, loss = 0.001342
grad_step = 000336, loss = 0.001350
grad_step = 000337, loss = 0.001363
grad_step = 000338, loss = 0.001374
grad_step = 000339, loss = 0.001390
grad_step = 000340, loss = 0.001397
grad_step = 000341, loss = 0.001410
grad_step = 000342, loss = 0.001401
grad_step = 000343, loss = 0.001391
grad_step = 000344, loss = 0.001355
grad_step = 000345, loss = 0.001324
grad_step = 000346, loss = 0.001307
grad_step = 000347, loss = 0.001313
grad_step = 000348, loss = 0.001330
grad_step = 000349, loss = 0.001338
grad_step = 000350, loss = 0.001336
grad_step = 000351, loss = 0.001320
grad_step = 000352, loss = 0.001309
grad_step = 000353, loss = 0.001303
grad_step = 000354, loss = 0.001302
grad_step = 000355, loss = 0.001305
grad_step = 000356, loss = 0.001311
grad_step = 000357, loss = 0.001321
grad_step = 000358, loss = 0.001329
grad_step = 000359, loss = 0.001338
grad_step = 000360, loss = 0.001341
grad_step = 000361, loss = 0.001346
grad_step = 000362, loss = 0.001348
grad_step = 000363, loss = 0.001356
grad_step = 000364, loss = 0.001352
grad_step = 000365, loss = 0.001349
grad_step = 000366, loss = 0.001324
grad_step = 000367, loss = 0.001304
grad_step = 000368, loss = 0.001289
grad_step = 000369, loss = 0.001288
grad_step = 000370, loss = 0.001291
grad_step = 000371, loss = 0.001289
grad_step = 000372, loss = 0.001285
grad_step = 000373, loss = 0.001285
grad_step = 000374, loss = 0.001290
grad_step = 000375, loss = 0.001295
grad_step = 000376, loss = 0.001297
grad_step = 000377, loss = 0.001296
grad_step = 000378, loss = 0.001292
grad_step = 000379, loss = 0.001286
grad_step = 000380, loss = 0.001282
grad_step = 000381, loss = 0.001281
grad_step = 000382, loss = 0.001288
grad_step = 000383, loss = 0.001303
grad_step = 000384, loss = 0.001332
grad_step = 000385, loss = 0.001354
grad_step = 000386, loss = 0.001390
grad_step = 000387, loss = 0.001383
grad_step = 000388, loss = 0.001395
grad_step = 000389, loss = 0.001405
grad_step = 000390, loss = 0.001436
grad_step = 000391, loss = 0.001397
grad_step = 000392, loss = 0.001317
grad_step = 000393, loss = 0.001266
grad_step = 000394, loss = 0.001289
grad_step = 000395, loss = 0.001314
grad_step = 000396, loss = 0.001293
grad_step = 000397, loss = 0.001284
grad_step = 000398, loss = 0.001297
grad_step = 000399, loss = 0.001300
grad_step = 000400, loss = 0.001264
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001249
grad_step = 000402, loss = 0.001269
grad_step = 000403, loss = 0.001276
grad_step = 000404, loss = 0.001264
grad_step = 000405, loss = 0.001253
grad_step = 000406, loss = 0.001259
grad_step = 000407, loss = 0.001265
grad_step = 000408, loss = 0.001257
grad_step = 000409, loss = 0.001244
grad_step = 000410, loss = 0.001242
grad_step = 000411, loss = 0.001247
grad_step = 000412, loss = 0.001246
grad_step = 000413, loss = 0.001239
grad_step = 000414, loss = 0.001234
grad_step = 000415, loss = 0.001232
grad_step = 000416, loss = 0.001236
grad_step = 000417, loss = 0.001240
grad_step = 000418, loss = 0.001241
grad_step = 000419, loss = 0.001242
grad_step = 000420, loss = 0.001251
grad_step = 000421, loss = 0.001271
grad_step = 000422, loss = 0.001322
grad_step = 000423, loss = 0.001379
grad_step = 000424, loss = 0.001489
grad_step = 000425, loss = 0.001549
grad_step = 000426, loss = 0.001618
grad_step = 000427, loss = 0.001501
grad_step = 000428, loss = 0.001332
grad_step = 000429, loss = 0.001221
grad_step = 000430, loss = 0.001296
grad_step = 000431, loss = 0.001389
grad_step = 000432, loss = 0.001325
grad_step = 000433, loss = 0.001255
grad_step = 000434, loss = 0.001252
grad_step = 000435, loss = 0.001270
grad_step = 000436, loss = 0.001290
grad_step = 000437, loss = 0.001273
grad_step = 000438, loss = 0.001232
grad_step = 000439, loss = 0.001212
grad_step = 000440, loss = 0.001243
grad_step = 000441, loss = 0.001272
grad_step = 000442, loss = 0.001229
grad_step = 000443, loss = 0.001202
grad_step = 000444, loss = 0.001221
grad_step = 000445, loss = 0.001229
grad_step = 000446, loss = 0.001211
grad_step = 000447, loss = 0.001194
grad_step = 000448, loss = 0.001201
grad_step = 000449, loss = 0.001212
grad_step = 000450, loss = 0.001199
grad_step = 000451, loss = 0.001188
grad_step = 000452, loss = 0.001193
grad_step = 000453, loss = 0.001198
grad_step = 000454, loss = 0.001193
grad_step = 000455, loss = 0.001183
grad_step = 000456, loss = 0.001179
grad_step = 000457, loss = 0.001184
grad_step = 000458, loss = 0.001186
grad_step = 000459, loss = 0.001182
grad_step = 000460, loss = 0.001175
grad_step = 000461, loss = 0.001171
grad_step = 000462, loss = 0.001172
grad_step = 000463, loss = 0.001174
grad_step = 000464, loss = 0.001173
grad_step = 000465, loss = 0.001169
grad_step = 000466, loss = 0.001164
grad_step = 000467, loss = 0.001162
grad_step = 000468, loss = 0.001162
grad_step = 000469, loss = 0.001162
grad_step = 000470, loss = 0.001159
grad_step = 000471, loss = 0.001156
grad_step = 000472, loss = 0.001154
grad_step = 000473, loss = 0.001153
grad_step = 000474, loss = 0.001153
grad_step = 000475, loss = 0.001152
grad_step = 000476, loss = 0.001152
grad_step = 000477, loss = 0.001153
grad_step = 000478, loss = 0.001159
grad_step = 000479, loss = 0.001173
grad_step = 000480, loss = 0.001202
grad_step = 000481, loss = 0.001267
grad_step = 000482, loss = 0.001361
grad_step = 000483, loss = 0.001546
grad_step = 000484, loss = 0.001651
grad_step = 000485, loss = 0.001731
grad_step = 000486, loss = 0.001424
grad_step = 000487, loss = 0.001165
grad_step = 000488, loss = 0.001208
grad_step = 000489, loss = 0.001375
grad_step = 000490, loss = 0.001353
grad_step = 000491, loss = 0.001182
grad_step = 000492, loss = 0.001192
grad_step = 000493, loss = 0.001269
grad_step = 000494, loss = 0.001249
grad_step = 000495, loss = 0.001214
grad_step = 000496, loss = 0.001162
grad_step = 000497, loss = 0.001147
grad_step = 000498, loss = 0.001209
grad_step = 000499, loss = 0.001203
grad_step = 000500, loss = 0.001131
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001116
Finished.
  ('#### Inference Need return ypred, ytrue #########################',) 
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 05:00:53.080711
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.208551
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 05:00:53.087056
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 0.0978937
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 05:00:53.094495
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.142657
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 05:00:53.099522
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -0.48753
metric_name                                             r2_score
Name: 3, dtype: object,) 
  ("### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10},) 
  ('#### Setup Model   ##############################################',) 
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
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f739b4a5400> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 344165.0625
Epoch 2/10

1/1 [==============================] - 0s 96ms/step - loss: 171595.9531
Epoch 3/10

1/1 [==============================] - 0s 90ms/step - loss: 80330.3906
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 38539.5039
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 20966.1660
Epoch 6/10

1/1 [==============================] - 0s 94ms/step - loss: 12833.6982
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 8640.2275
Epoch 8/10

1/1 [==============================] - 0s 89ms/step - loss: 6266.8174
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 4837.5449
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 3933.8506
  ('#### Inference Need return ypred, ytrue #########################',) 
[[ -1.4635433    1.1876853   -0.8082291    2.6067147    0.3164361
   -2.0742514    0.6291375    0.53525263  -0.49898848   1.3204596
   -0.19995672  -1.7672267    2.115044    -1.1994766    1.4068344
    1.18142     -0.4442228   -0.932369    -2.3141174    2.6774442
   -1.2401394   -1.137672     2.7230685   -1.4516879    1.8220716
   -0.9280209    0.42292064  -1.8691981    1.5512238    1.1485026
    0.0994733    0.14655042  -1.9146174    0.1763131   -0.4013077
   -3.3456268   -0.25450176  -2.119886     0.9616389   -0.16649348
    0.01850957  -1.0026282    0.649671     0.45841345   1.4781291
    0.56320035   1.5994492   -0.8762542    0.72197235   0.15437172
   -0.88954264  -1.0568745    0.6034009    0.61800766   0.8379539
   -0.22357008   1.576939    -0.8237216    2.2284176    0.24465534
    0.6826625   11.082959    11.592477     9.614552    11.647943
   12.539759     9.036611     8.092503     9.413362    12.282086
    9.43099     12.396363     9.70363     10.693563    10.439329
   10.353106    11.838712     8.900882    12.192323    12.727017
   11.444197    10.207959    13.315382    12.315236     9.347944
   12.090071    10.153883     9.947232    11.585592    10.244662
   11.369516     9.850787    12.509953    12.794936    10.529383
    9.38851     11.539283    10.406036    12.050612    12.199271
   12.979993    10.704975    10.4816065   11.841526    11.451892
   10.249674    13.938408    10.641624    11.833662    10.746332
   13.249147     9.494153    11.042538    11.593706    10.936155
   12.353912     8.993655    11.444117    12.164839    11.07207
   -1.7279705    1.0595282   -0.33561736   1.3728702   -1.850179
    1.3537049    1.6489046   -0.5309087   -0.7934115    1.3340923
   -0.45183522  -0.08040774  -0.3147587    0.4227597    0.4049985
   -0.73825896   0.94026214  -1.1559281   -0.47687227   0.8570063
   -0.26737517  -1.4654884    0.8605832   -0.2087782   -1.144192
    0.12773675  -1.8901001    0.64403147   2.2841089    2.1943698
    1.5747046   -0.40155303  -0.37572843   0.8439988   -1.8842276
    2.0703502    0.18851614  -3.1887107    0.03219585  -1.4789989
    2.299485     1.4329562    1.3369935    0.11024988   1.2313223
    1.257446     0.26431224   0.22355208   2.084994    -0.776481
    0.6940111    1.1119637   -1.0396714   -0.40869153  -0.82176363
    1.5424464    3.2469141    0.90890974  -1.0746137   -1.3449425
    0.8388951    1.4221656    0.55144906   1.8180857    2.2704473
    2.0881062    1.0670844    2.9112616    2.251766     2.7528315
    0.15643251   0.45944488   0.41638362   0.8187108    0.50954765
    3.161314     0.18610448   0.10506701   0.3426575    0.2991119
    2.0171878    0.5601075    2.518948     0.332918     2.0791588
    0.5004175    2.6121054    2.3160582    0.22505659   0.80911255
    0.6886399    1.0459313    0.08630961   3.1666737    2.1051555
    0.5748971    2.4823675    0.21297836   1.4976676    0.24279791
    1.3713099    0.96722096   0.6986822    1.0633619    0.8004298
    0.4067061    1.3291807    1.2465057    3.2285347    0.23462468
    1.0297881    1.0079722    0.2814967    0.4331149    0.30715805
    0.88580585   0.2765271    0.12751997   2.737167     2.25654
    0.68414134  11.888761     9.015501    11.841377    10.521225
   10.931205    11.35743      8.656127    13.327192    10.6130495
   11.681569    11.04922      9.96117      9.9241705   12.357224
    8.684119    11.7493925    8.672758    10.958449    11.857549
   14.08836     12.252702    13.087624     9.376293    11.968435
   11.172914    10.609531    11.46079     13.174921    11.200438
   11.155211    11.873562    13.055087    11.271837    10.723827
   10.607061    10.8842945   12.99134     11.2696      11.377509
    9.888305    13.221752     8.98032     12.747319     9.997281
    9.8656845   12.668756     8.447542    12.320328     9.966208
   14.076388    11.724059    13.243373     9.131163     8.183894
   12.372038    12.400549    11.147092    12.118164    12.846689
    1.0144179    2.389493     0.3411007    2.135538     1.0166657
    0.31319857   0.799397     0.17253137   0.89950424   2.8260179
    1.975642     1.8711663    1.938833     0.64485765   0.36517048
    0.54949087   0.29874116   2.1724792    2.738276     1.7736436
    0.19078934   0.4718399    1.8652055    3.0804973    0.86213696
    2.4425135    0.21207184   0.31491536   0.5341725    2.3586104
    2.978457     0.32689786   0.16383684   2.9189844    2.649472
    0.93812555   0.3820052    0.50486195   0.5049707    0.4621306
    0.6738558    1.3558973    1.4794273    0.7974854    0.6026819
    0.3938011    0.16857982   1.7716334    0.54142773   0.19106275
    0.41874284   0.20781374   2.3166504    0.6727488    0.7132076
    1.3834378    0.15552014   2.3342419    2.6316686    1.2741097
   -0.7226118   14.871103   -10.871371  ]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 05:01:01.534606
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    90.907
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 05:01:01.538483
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    8293.6
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 05:01:01.541693
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   90.4846
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 05:01:01.544946
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -741.751
metric_name                                             r2_score
Name: 7, dtype: object,) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon/fb_prophet.py'},) 
  ('#### Setup Model   ##############################################',) 
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f73446bff60> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  ({'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, FileNotFoundError(2, "File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist")) 
  ("### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, KeyError('model_uri',)) 
  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/',) 
  (                     date_run  ...            metric_name
0  2020-05-09 05:00:53.080711  ...    mean_absolute_error
1  2020-05-09 05:00:53.087056  ...     mean_squared_error
2  2020-05-09 05:00:53.094495  ...  median_absolute_error
3  2020-05-09 05:00:53.099522  ...               r2_score
4  2020-05-09 05:01:01.534606  ...    mean_absolute_error
5  2020-05-09 05:01:01.538483  ...     mean_squared_error
6  2020-05-09 05:01:01.541693  ...  median_absolute_error
7  2020-05-09 05:01:01.544946  ...               r2_score

[8 rows x 6 columns],) 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 114, in benchmark_run
    model_uri =  model_pars['model_uri']
KeyError: 'model_uri'

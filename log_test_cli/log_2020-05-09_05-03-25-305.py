  ('/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json',) 
  ('test_cli', 'GITHUB_REPOSITORT', 'GITHUB_SHA') 
  ('Running command', 'test_cli') 
  ('# Testing Command Line System  ',) 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/42ac725fa1a30b94674c28fe0e2a484d0db5dba3', 'url_branch_file': 'https://github.com/{repo}/blob/{branch}/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '42ac725fa1a30b94674c28fe0e2a484d0db5dba3', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/42ac725fa1a30b94674c28fe0e2a484d0db5dba3

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/42ac725fa1a30b94674c28fe0e2a484d0db5dba3

 ************************************************************************************************************************
Using : /home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md
['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n']





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
  (<mlmodels.model_tf.1_lstm.Model object at 0x7fc4ef5897f0>,) 
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
 [-0.18892388 -0.05083193  0.15371223  0.087479    0.08084415 -0.0935822 ]
 [-0.07719564  0.05479869 -0.06161483 -0.03874354  0.02009194  0.10177929]
 [-0.04283978  0.06410272 -0.01986745  0.00638747  0.1793233   0.00172628]
 [ 0.40485853 -0.22371332  0.33338061  0.15821755  0.2259967   0.03193437]
 [-0.67296189  0.00946994  0.15071093  0.2966924  -0.14406231 -0.11724709]
 [ 0.00903176 -0.07221445  0.76998675  0.77078849 -0.30929321  1.06763494]
 [ 0.67614311 -0.59959418  1.08344615  0.68724483  0.6880185   0.07646734]
 [-0.19075949  0.48198512  0.09302707 -0.10900582  0.23736288  0.05062161]
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
{'loss': 0.4731381982564926, 'loss_history': []}
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
{'loss': 0.450006689876318, 'loss_history': []}
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
[32m[I 2020-05-09 05:03:54,376][0m Finished trial#0 resulted in value: 0.29874666780233383. Current best value is 0.29874666780233383 with parameters: {'learning_rate': 0.0018117539483271236, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-09 05:03:55,718][0m Finished trial#1 resulted in value: 0.4341766834259033. Current best value is 0.29874666780233383 with parameters: {'learning_rate': 0.0018117539483271236, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f58e80992b0> <class 'mlmodels.model_gluon.fb_prophet.Model'>
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f58d74849b0> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357549.0312
Epoch 2/10

1/1 [==============================] - 0s 111ms/step - loss: 296330.4062
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 215037.2969
Epoch 4/10

1/1 [==============================] - 0s 106ms/step - loss: 131341.3438
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 73464.3828
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 41698.0820
Epoch 7/10

1/1 [==============================] - 0s 100ms/step - loss: 25092.5625
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 16034.9336
Epoch 9/10

1/1 [==============================] - 0s 95ms/step - loss: 10905.3350
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 7859.6963
  ('#### Inference Need return ypred, ytrue #########################',) 
[[-3.12094092e-01  8.85853171e-03 -1.26218414e+00 -3.74329388e-02
   3.16333890e-01  3.51158977e-02  1.12463999e+00  8.24735999e-01
  -6.49573207e-02 -9.12376463e-01  1.74557972e+00 -9.73117709e-01
  -1.15172994e+00 -2.21800208e+00 -8.45380425e-02  6.55190349e-01
   3.73093843e-01 -1.01321891e-01  9.31479037e-01 -7.29096293e-01
   7.96296000e-01  7.16815233e-01 -1.20228529e+00  1.12698162e+00
  -4.45163757e-01 -4.52814609e-01 -9.10368800e-01  1.14424658e+00
   1.46809316e+00 -1.35280812e+00  8.97339106e-01 -3.78721088e-01
  -1.51813972e+00 -9.57366288e-01  1.57819080e+00  3.26028556e-01
  -7.59478509e-02 -3.78665984e-01  9.06380773e-01 -5.31113327e-01
  -7.20812798e-01 -1.18416041e-01  1.07481968e+00 -9.28607166e-01
   4.71691102e-01  1.10166013e+00  1.46544233e-01 -7.03493893e-01
  -7.38324106e-01 -6.26543760e-01 -1.55985117e+00  6.69671535e-01
   6.37743115e-01  1.11200428e+00 -5.06462812e-01  1.35146856e+00
  -1.32865095e+00 -7.41119146e-01 -5.56533277e-01 -3.89729440e-01
  -3.20156872e-01 -4.03374970e-01 -3.39536190e-01 -1.02311170e+00
  -4.75086272e-01  1.62549543e+00  1.00665438e+00 -2.65336037e-01
  -4.33809340e-01 -9.97133195e-01 -1.52262771e+00  6.78205937e-02
   5.93854845e-01  6.57759190e-01  1.29375958e+00  5.96203446e-01
  -2.00450826e+00  6.07696772e-02 -1.21767485e+00 -6.38129234e-01
   9.67649579e-01  9.24115121e-01  1.68631792e+00  7.67673254e-02
   6.35898888e-01  5.48113942e-01 -1.19477510e-01 -5.07238865e-01
   2.32643932e-02  1.62011847e-01 -3.64932299e-01  3.57811630e-01
   6.71480969e-02 -2.66739547e-01  8.82041156e-01  1.55844963e+00
   1.38087928e-01  1.80925131e-01  5.82622528e-01 -4.02858853e-01
  -5.72386421e-02 -9.40539479e-01  1.02344036e+00 -2.27935910e-01
  -7.63160408e-01 -4.49967980e-01  1.48029113e+00 -5.21108687e-01
  -1.81750059e+00 -4.31285679e-01 -1.23891771e+00 -1.92611933e+00
  -7.99447656e-01  4.89665151e-01  9.39745486e-01  6.49325073e-01
  -1.04635859e+00 -3.87370810e-02  3.08388650e-01 -1.96300685e-01
   2.79127866e-01  7.36374998e+00  7.96099758e+00  8.89903641e+00
   8.53776073e+00  5.73878193e+00  6.70785236e+00  5.87255526e+00
   8.04751301e+00  8.35083961e+00  6.14310312e+00  6.81443739e+00
   6.88425112e+00  8.00777912e+00  9.05643940e+00  7.76194334e+00
   8.79174614e+00  7.31639624e+00  5.78268099e+00  7.36091042e+00
   7.66554070e+00  7.59710598e+00  7.67121458e+00  8.05671883e+00
   7.12623405e+00  6.73061752e+00  5.79317093e+00  8.59305096e+00
   7.98338795e+00  8.33560181e+00  6.03910828e+00  7.79045630e+00
   6.81633520e+00  6.38499117e+00  7.44529724e+00  7.71663570e+00
   7.44789267e+00  9.12309933e+00  6.53451204e+00  7.55240917e+00
   6.47163868e+00  7.87398529e+00  6.08691311e+00  6.25559425e+00
   6.72986507e+00  7.77179098e+00  7.40446186e+00  7.93438530e+00
   8.12116814e+00  8.48398781e+00  7.10086536e+00  7.77175093e+00
   6.72684622e+00  6.73787451e+00  6.89149380e+00  8.50541782e+00
   6.64871788e+00  7.84752798e+00  6.18764114e+00  8.34715366e+00
   1.49813557e+00  1.00384593e+00  1.09692156e-01  2.46971309e-01
   4.70464706e-01  8.30669999e-01  7.79573381e-01  3.30914378e-01
   1.79538929e+00  1.07279778e+00  2.33256388e+00  1.33085561e+00
   6.13469481e-01  3.14054728e-01  2.06602955e+00  1.05388618e+00
   3.28797674e+00  2.94925642e+00  6.57475829e-01  2.27459717e+00
   2.18449926e+00  1.28098607e+00  2.56835890e+00  5.46477795e-01
   3.47224355e-01  8.19156885e-01  7.09758699e-01  4.11820650e-01
   2.26710463e+00  2.25913763e-01  8.33295345e-01  8.65648627e-01
   1.23212099e+00  5.03878653e-01  5.74731350e-01  8.86464179e-01
   3.49752009e-01  1.05863094e-01  2.79206109e+00  9.59387243e-01
   2.55707312e+00  2.38776398e+00  1.36996150e+00  1.18352008e+00
   3.99485707e-01  2.10710239e+00  3.74876618e-01  3.09182525e-01
   3.68466854e-01  9.25929308e-01  7.56250799e-01  3.92210841e-01
   6.88827991e-01  5.16640007e-01  1.65235317e+00  1.72842205e-01
   2.97651649e-01  4.82992053e-01  8.74470711e-01  2.03663158e+00
   4.81780648e-01  4.88386989e-01  9.75382864e-01  1.82044435e+00
   3.50783467e-01  1.60074794e+00  2.00242996e+00  9.27028179e-01
   9.11951661e-01  2.13050890e+00  9.55753028e-01  1.53473163e+00
   2.39222705e-01  1.88254654e-01  1.94485199e+00  1.80196559e+00
   8.34779859e-01  2.92552710e-01  2.21327209e+00  8.27304542e-01
   7.67189264e-01  1.95802307e+00  7.81281531e-01  1.70893526e+00
   2.69188976e+00  3.35470319e-01  4.74215746e-01  2.03959179e+00
   5.41847765e-01  8.64356995e-01  5.01992583e-01  4.53050971e-01
   2.01659846e+00  3.80678833e-01  5.05014122e-01  2.09919810e+00
   5.85068703e-01  1.72668338e-01  3.26726735e-01  2.11531699e-01
   1.92785525e+00  8.01628232e-01  4.10208404e-01  3.08553398e-01
   1.91691923e+00  2.42047906e-01  1.85404921e+00  7.41720319e-01
   1.51707721e+00  3.62812877e-01  1.51724100e+00  4.39373016e-01
   1.68914127e+00  4.66831386e-01  3.81430578e+00  1.75296724e-01
   1.53485692e+00  3.32579136e-01  1.16068041e+00  4.67337370e-01
   5.37707806e-02  6.54835558e+00  6.26907110e+00  9.19026470e+00
   6.08342791e+00  8.34329319e+00  8.26863670e+00  7.87612915e+00
   7.99434042e+00  7.63095760e+00  8.06977654e+00  8.30966759e+00
   8.20113182e+00  6.38221550e+00  8.01371861e+00  7.12031317e+00
   8.59467411e+00  7.02062321e+00  8.39921284e+00  7.80493069e+00
   8.52208996e+00  8.61114407e+00  7.32430172e+00  9.74759674e+00
   8.27529049e+00  8.02404308e+00  8.12084389e+00  7.90118074e+00
   9.49526691e+00  7.23487473e+00  7.62967587e+00  7.49433851e+00
   6.78031206e+00  7.40430069e+00  8.47346878e+00  9.69402122e+00
   7.10551214e+00  7.78101492e+00  7.82050514e+00  7.54196835e+00
   7.13104963e+00  7.52315950e+00  8.82420921e+00  8.81175137e+00
   7.60603333e+00  8.97235012e+00  7.53474808e+00  9.48355293e+00
   6.34750986e+00  7.59230614e+00  8.29530907e+00  7.05841827e+00
   6.98023987e+00  6.95166111e+00  7.22362328e+00  7.96423578e+00
   8.46031094e+00  7.67531586e+00  7.66519880e+00  8.08915997e+00
  -7.95570374e+00 -9.34842300e+00  5.78585482e+00]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 05:04:09.108907
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.9266
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 05:04:09.113023
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9031.31
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 05:04:09.116449
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.4562
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 05:04:09.119570
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -807.818
metric_name                                             r2_score
Name: 3, dtype: object,) 
  ("### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256},) 
  ('#### Setup Model   ##############################################',) 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140019210910184
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140016700288192
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140016700288696
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140016699904240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140016699904744
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140016699905248
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f58e20de400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.630298
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.598738
grad_step = 000002, loss = 0.577811
grad_step = 000003, loss = 0.557320
grad_step = 000004, loss = 0.535556
grad_step = 000005, loss = 0.516556
grad_step = 000006, loss = 0.504832
grad_step = 000007, loss = 0.495692
grad_step = 000008, loss = 0.477842
grad_step = 000009, loss = 0.456796
grad_step = 000010, loss = 0.439984
grad_step = 000011, loss = 0.428435
grad_step = 000012, loss = 0.417330
grad_step = 000013, loss = 0.404615
grad_step = 000014, loss = 0.389191
grad_step = 000015, loss = 0.374419
grad_step = 000016, loss = 0.363458
grad_step = 000017, loss = 0.355096
grad_step = 000018, loss = 0.344124
grad_step = 000019, loss = 0.330538
grad_step = 000020, loss = 0.318276
grad_step = 000021, loss = 0.308454
grad_step = 000022, loss = 0.298582
grad_step = 000023, loss = 0.287189
grad_step = 000024, loss = 0.275304
grad_step = 000025, loss = 0.264104
grad_step = 000026, loss = 0.253579
grad_step = 000027, loss = 0.243331
grad_step = 000028, loss = 0.233089
grad_step = 000029, loss = 0.223153
grad_step = 000030, loss = 0.213392
grad_step = 000031, loss = 0.203808
grad_step = 000032, loss = 0.194773
grad_step = 000033, loss = 0.185927
grad_step = 000034, loss = 0.176902
grad_step = 000035, loss = 0.168051
grad_step = 000036, loss = 0.160035
grad_step = 000037, loss = 0.152539
grad_step = 000038, loss = 0.144768
grad_step = 000039, loss = 0.137250
grad_step = 000040, loss = 0.130521
grad_step = 000041, loss = 0.124028
grad_step = 000042, loss = 0.117402
grad_step = 000043, loss = 0.111093
grad_step = 000044, loss = 0.105418
grad_step = 000045, loss = 0.099866
grad_step = 000046, loss = 0.094390
grad_step = 000047, loss = 0.089302
grad_step = 000048, loss = 0.084521
grad_step = 000049, loss = 0.079898
grad_step = 000050, loss = 0.075495
grad_step = 000051, loss = 0.071383
grad_step = 000052, loss = 0.067517
grad_step = 000053, loss = 0.063757
grad_step = 000054, loss = 0.060142
grad_step = 000055, loss = 0.056813
grad_step = 000056, loss = 0.053688
grad_step = 000057, loss = 0.050693
grad_step = 000058, loss = 0.047963
grad_step = 000059, loss = 0.045406
grad_step = 000060, loss = 0.042910
grad_step = 000061, loss = 0.040618
grad_step = 000062, loss = 0.038469
grad_step = 000063, loss = 0.036396
grad_step = 000064, loss = 0.034498
grad_step = 000065, loss = 0.032731
grad_step = 000066, loss = 0.031036
grad_step = 000067, loss = 0.029447
grad_step = 000068, loss = 0.027960
grad_step = 000069, loss = 0.026549
grad_step = 000070, loss = 0.025215
grad_step = 000071, loss = 0.023987
grad_step = 000072, loss = 0.022818
grad_step = 000073, loss = 0.021706
grad_step = 000074, loss = 0.020673
grad_step = 000075, loss = 0.019673
grad_step = 000076, loss = 0.018732
grad_step = 000077, loss = 0.017852
grad_step = 000078, loss = 0.017003
grad_step = 000079, loss = 0.016212
grad_step = 000080, loss = 0.015447
grad_step = 000081, loss = 0.014721
grad_step = 000082, loss = 0.014033
grad_step = 000083, loss = 0.013377
grad_step = 000084, loss = 0.012746
grad_step = 000085, loss = 0.012150
grad_step = 000086, loss = 0.011583
grad_step = 000087, loss = 0.011044
grad_step = 000088, loss = 0.010546
grad_step = 000089, loss = 0.010072
grad_step = 000090, loss = 0.009625
grad_step = 000091, loss = 0.009147
grad_step = 000092, loss = 0.008665
grad_step = 000093, loss = 0.008233
grad_step = 000094, loss = 0.007867
grad_step = 000095, loss = 0.007521
grad_step = 000096, loss = 0.007150
grad_step = 000097, loss = 0.006778
grad_step = 000098, loss = 0.006451
grad_step = 000099, loss = 0.006171
grad_step = 000100, loss = 0.005899
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.005611
grad_step = 000102, loss = 0.005331
grad_step = 000103, loss = 0.005087
grad_step = 000104, loss = 0.004874
grad_step = 000105, loss = 0.004670
grad_step = 000106, loss = 0.004459
grad_step = 000107, loss = 0.004253
grad_step = 000108, loss = 0.004068
grad_step = 000109, loss = 0.003908
grad_step = 000110, loss = 0.003762
grad_step = 000111, loss = 0.003621
grad_step = 000112, loss = 0.003481
grad_step = 000113, loss = 0.003345
grad_step = 000114, loss = 0.003221
grad_step = 000115, loss = 0.003109
grad_step = 000116, loss = 0.003011
grad_step = 000117, loss = 0.002924
grad_step = 000118, loss = 0.002844
grad_step = 000119, loss = 0.002773
grad_step = 000120, loss = 0.002707
grad_step = 000121, loss = 0.002650
grad_step = 000122, loss = 0.002594
grad_step = 000123, loss = 0.002544
grad_step = 000124, loss = 0.002488
grad_step = 000125, loss = 0.002434
grad_step = 000126, loss = 0.002375
grad_step = 000127, loss = 0.002323
grad_step = 000128, loss = 0.002279
grad_step = 000129, loss = 0.002245
grad_step = 000130, loss = 0.002220
grad_step = 000131, loss = 0.002201
grad_step = 000132, loss = 0.002190
grad_step = 000133, loss = 0.002183
grad_step = 000134, loss = 0.002187
grad_step = 000135, loss = 0.002191
grad_step = 000136, loss = 0.002199
grad_step = 000137, loss = 0.002177
grad_step = 000138, loss = 0.002141
grad_step = 000139, loss = 0.002087
grad_step = 000140, loss = 0.002052
grad_step = 000141, loss = 0.002047
grad_step = 000142, loss = 0.002063
grad_step = 000143, loss = 0.002081
grad_step = 000144, loss = 0.002078
grad_step = 000145, loss = 0.002059
grad_step = 000146, loss = 0.002029
grad_step = 000147, loss = 0.002008
grad_step = 000148, loss = 0.002005
grad_step = 000149, loss = 0.002014
grad_step = 000150, loss = 0.002026
grad_step = 000151, loss = 0.002029
grad_step = 000152, loss = 0.002025
grad_step = 000153, loss = 0.002010
grad_step = 000154, loss = 0.001994
grad_step = 000155, loss = 0.001981
grad_step = 000156, loss = 0.001973
grad_step = 000157, loss = 0.001970
grad_step = 000158, loss = 0.001966
grad_step = 000159, loss = 0.001962
grad_step = 000160, loss = 0.001960
grad_step = 000161, loss = 0.001958
grad_step = 000162, loss = 0.001961
grad_step = 000163, loss = 0.001974
grad_step = 000164, loss = 0.002011
grad_step = 000165, loss = 0.002110
grad_step = 000166, loss = 0.002248
grad_step = 000167, loss = 0.002421
grad_step = 000168, loss = 0.002207
grad_step = 000169, loss = 0.001959
grad_step = 000170, loss = 0.001984
grad_step = 000171, loss = 0.002141
grad_step = 000172, loss = 0.002097
grad_step = 000173, loss = 0.001923
grad_step = 000174, loss = 0.001974
grad_step = 000175, loss = 0.002078
grad_step = 000176, loss = 0.001972
grad_step = 000177, loss = 0.001898
grad_step = 000178, loss = 0.001975
grad_step = 000179, loss = 0.001986
grad_step = 000180, loss = 0.001909
grad_step = 000181, loss = 0.001895
grad_step = 000182, loss = 0.001935
grad_step = 000183, loss = 0.001938
grad_step = 000184, loss = 0.001904
grad_step = 000185, loss = 0.001883
grad_step = 000186, loss = 0.001870
grad_step = 000187, loss = 0.001885
grad_step = 000188, loss = 0.001930
grad_step = 000189, loss = 0.001914
grad_step = 000190, loss = 0.001874
grad_step = 000191, loss = 0.001841
grad_step = 000192, loss = 0.001843
grad_step = 000193, loss = 0.001854
grad_step = 000194, loss = 0.001885
grad_step = 000195, loss = 0.001967
grad_step = 000196, loss = 0.002012
grad_step = 000197, loss = 0.002038
grad_step = 000198, loss = 0.001889
grad_step = 000199, loss = 0.001807
grad_step = 000200, loss = 0.001876
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001905
grad_step = 000202, loss = 0.001842
grad_step = 000203, loss = 0.001799
grad_step = 000204, loss = 0.001847
grad_step = 000205, loss = 0.001885
grad_step = 000206, loss = 0.001825
grad_step = 000207, loss = 0.001788
grad_step = 000208, loss = 0.001805
grad_step = 000209, loss = 0.001833
grad_step = 000210, loss = 0.001840
grad_step = 000211, loss = 0.001800
grad_step = 000212, loss = 0.001776
grad_step = 000213, loss = 0.001781
grad_step = 000214, loss = 0.001799
grad_step = 000215, loss = 0.001817
grad_step = 000216, loss = 0.001811
grad_step = 000217, loss = 0.001800
grad_step = 000218, loss = 0.001776
grad_step = 000219, loss = 0.001762
grad_step = 000220, loss = 0.001764
grad_step = 000221, loss = 0.001776
grad_step = 000222, loss = 0.001793
grad_step = 000223, loss = 0.001809
grad_step = 000224, loss = 0.001833
grad_step = 000225, loss = 0.001842
grad_step = 000226, loss = 0.001850
grad_step = 000227, loss = 0.001822
grad_step = 000228, loss = 0.001785
grad_step = 000229, loss = 0.001754
grad_step = 000230, loss = 0.001752
grad_step = 000231, loss = 0.001770
grad_step = 000232, loss = 0.001784
grad_step = 000233, loss = 0.001779
grad_step = 000234, loss = 0.001760
grad_step = 000235, loss = 0.001745
grad_step = 000236, loss = 0.001742
grad_step = 000237, loss = 0.001751
grad_step = 000238, loss = 0.001763
grad_step = 000239, loss = 0.001766
grad_step = 000240, loss = 0.001764
grad_step = 000241, loss = 0.001754
grad_step = 000242, loss = 0.001744
grad_step = 000243, loss = 0.001736
grad_step = 000244, loss = 0.001732
grad_step = 000245, loss = 0.001732
grad_step = 000246, loss = 0.001735
grad_step = 000247, loss = 0.001740
grad_step = 000248, loss = 0.001747
grad_step = 000249, loss = 0.001759
grad_step = 000250, loss = 0.001775
grad_step = 000251, loss = 0.001802
grad_step = 000252, loss = 0.001822
grad_step = 000253, loss = 0.001838
grad_step = 000254, loss = 0.001807
grad_step = 000255, loss = 0.001761
grad_step = 000256, loss = 0.001727
grad_step = 000257, loss = 0.001733
grad_step = 000258, loss = 0.001757
grad_step = 000259, loss = 0.001760
grad_step = 000260, loss = 0.001744
grad_step = 000261, loss = 0.001725
grad_step = 000262, loss = 0.001723
grad_step = 000263, loss = 0.001733
grad_step = 000264, loss = 0.001740
grad_step = 000265, loss = 0.001737
grad_step = 000266, loss = 0.001724
grad_step = 000267, loss = 0.001714
grad_step = 000268, loss = 0.001713
grad_step = 000269, loss = 0.001719
grad_step = 000270, loss = 0.001726
grad_step = 000271, loss = 0.001729
grad_step = 000272, loss = 0.001734
grad_step = 000273, loss = 0.001741
grad_step = 000274, loss = 0.001751
grad_step = 000275, loss = 0.001754
grad_step = 000276, loss = 0.001758
grad_step = 000277, loss = 0.001751
grad_step = 000278, loss = 0.001741
grad_step = 000279, loss = 0.001725
grad_step = 000280, loss = 0.001711
grad_step = 000281, loss = 0.001704
grad_step = 000282, loss = 0.001706
grad_step = 000283, loss = 0.001714
grad_step = 000284, loss = 0.001718
grad_step = 000285, loss = 0.001715
grad_step = 000286, loss = 0.001706
grad_step = 000287, loss = 0.001698
grad_step = 000288, loss = 0.001696
grad_step = 000289, loss = 0.001699
grad_step = 000290, loss = 0.001702
grad_step = 000291, loss = 0.001704
grad_step = 000292, loss = 0.001707
grad_step = 000293, loss = 0.001711
grad_step = 000294, loss = 0.001721
grad_step = 000295, loss = 0.001732
grad_step = 000296, loss = 0.001751
grad_step = 000297, loss = 0.001765
grad_step = 000298, loss = 0.001784
grad_step = 000299, loss = 0.001776
grad_step = 000300, loss = 0.001761
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001724
grad_step = 000302, loss = 0.001700
grad_step = 000303, loss = 0.001696
grad_step = 000304, loss = 0.001705
grad_step = 000305, loss = 0.001715
grad_step = 000306, loss = 0.001717
grad_step = 000307, loss = 0.001712
grad_step = 000308, loss = 0.001700
grad_step = 000309, loss = 0.001687
grad_step = 000310, loss = 0.001683
grad_step = 000311, loss = 0.001690
grad_step = 000312, loss = 0.001698
grad_step = 000313, loss = 0.001700
grad_step = 000314, loss = 0.001692
grad_step = 000315, loss = 0.001681
grad_step = 000316, loss = 0.001676
grad_step = 000317, loss = 0.001678
grad_step = 000318, loss = 0.001682
grad_step = 000319, loss = 0.001682
grad_step = 000320, loss = 0.001678
grad_step = 000321, loss = 0.001673
grad_step = 000322, loss = 0.001670
grad_step = 000323, loss = 0.001671
grad_step = 000324, loss = 0.001673
grad_step = 000325, loss = 0.001673
grad_step = 000326, loss = 0.001671
grad_step = 000327, loss = 0.001668
grad_step = 000328, loss = 0.001665
grad_step = 000329, loss = 0.001665
grad_step = 000330, loss = 0.001666
grad_step = 000331, loss = 0.001666
grad_step = 000332, loss = 0.001667
grad_step = 000333, loss = 0.001667
grad_step = 000334, loss = 0.001670
grad_step = 000335, loss = 0.001678
grad_step = 000336, loss = 0.001698
grad_step = 000337, loss = 0.001733
grad_step = 000338, loss = 0.001809
grad_step = 000339, loss = 0.001889
grad_step = 000340, loss = 0.002029
grad_step = 000341, loss = 0.001976
grad_step = 000342, loss = 0.001856
grad_step = 000343, loss = 0.001723
grad_step = 000344, loss = 0.001751
grad_step = 000345, loss = 0.001802
grad_step = 000346, loss = 0.001727
grad_step = 000347, loss = 0.001703
grad_step = 000348, loss = 0.001768
grad_step = 000349, loss = 0.001772
grad_step = 000350, loss = 0.001716
grad_step = 000351, loss = 0.001654
grad_step = 000352, loss = 0.001684
grad_step = 000353, loss = 0.001739
grad_step = 000354, loss = 0.001707
grad_step = 000355, loss = 0.001657
grad_step = 000356, loss = 0.001653
grad_step = 000357, loss = 0.001686
grad_step = 000358, loss = 0.001701
grad_step = 000359, loss = 0.001666
grad_step = 000360, loss = 0.001640
grad_step = 000361, loss = 0.001650
grad_step = 000362, loss = 0.001669
grad_step = 000363, loss = 0.001667
grad_step = 000364, loss = 0.001645
grad_step = 000365, loss = 0.001639
grad_step = 000366, loss = 0.001649
grad_step = 000367, loss = 0.001653
grad_step = 000368, loss = 0.001643
grad_step = 000369, loss = 0.001631
grad_step = 000370, loss = 0.001634
grad_step = 000371, loss = 0.001642
grad_step = 000372, loss = 0.001640
grad_step = 000373, loss = 0.001631
grad_step = 000374, loss = 0.001628
grad_step = 000375, loss = 0.001631
grad_step = 000376, loss = 0.001632
grad_step = 000377, loss = 0.001627
grad_step = 000378, loss = 0.001622
grad_step = 000379, loss = 0.001621
grad_step = 000380, loss = 0.001623
grad_step = 000381, loss = 0.001622
grad_step = 000382, loss = 0.001618
grad_step = 000383, loss = 0.001615
grad_step = 000384, loss = 0.001615
grad_step = 000385, loss = 0.001616
grad_step = 000386, loss = 0.001615
grad_step = 000387, loss = 0.001612
grad_step = 000388, loss = 0.001609
grad_step = 000389, loss = 0.001608
grad_step = 000390, loss = 0.001608
grad_step = 000391, loss = 0.001607
grad_step = 000392, loss = 0.001606
grad_step = 000393, loss = 0.001603
grad_step = 000394, loss = 0.001601
grad_step = 000395, loss = 0.001600
grad_step = 000396, loss = 0.001599
grad_step = 000397, loss = 0.001599
grad_step = 000398, loss = 0.001598
grad_step = 000399, loss = 0.001598
grad_step = 000400, loss = 0.001598
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001599
grad_step = 000402, loss = 0.001603
grad_step = 000403, loss = 0.001613
grad_step = 000404, loss = 0.001627
grad_step = 000405, loss = 0.001658
grad_step = 000406, loss = 0.001694
grad_step = 000407, loss = 0.001765
grad_step = 000408, loss = 0.001809
grad_step = 000409, loss = 0.001871
grad_step = 000410, loss = 0.001827
grad_step = 000411, loss = 0.001773
grad_step = 000412, loss = 0.001709
grad_step = 000413, loss = 0.001649
grad_step = 000414, loss = 0.001614
grad_step = 000415, loss = 0.001603
grad_step = 000416, loss = 0.001636
grad_step = 000417, loss = 0.001678
grad_step = 000418, loss = 0.001665
grad_step = 000419, loss = 0.001611
grad_step = 000420, loss = 0.001563
grad_step = 000421, loss = 0.001568
grad_step = 000422, loss = 0.001602
grad_step = 000423, loss = 0.001611
grad_step = 000424, loss = 0.001587
grad_step = 000425, loss = 0.001563
grad_step = 000426, loss = 0.001568
grad_step = 000427, loss = 0.001596
grad_step = 000428, loss = 0.001606
grad_step = 000429, loss = 0.001604
grad_step = 000430, loss = 0.001596
grad_step = 000431, loss = 0.001613
grad_step = 000432, loss = 0.001634
grad_step = 000433, loss = 0.001645
grad_step = 000434, loss = 0.001619
grad_step = 000435, loss = 0.001584
grad_step = 000436, loss = 0.001552
grad_step = 000437, loss = 0.001541
grad_step = 000438, loss = 0.001543
grad_step = 000439, loss = 0.001555
grad_step = 000440, loss = 0.001567
grad_step = 000441, loss = 0.001565
grad_step = 000442, loss = 0.001554
grad_step = 000443, loss = 0.001535
grad_step = 000444, loss = 0.001526
grad_step = 000445, loss = 0.001531
grad_step = 000446, loss = 0.001540
grad_step = 000447, loss = 0.001546
grad_step = 000448, loss = 0.001543
grad_step = 000449, loss = 0.001536
grad_step = 000450, loss = 0.001527
grad_step = 000451, loss = 0.001521
grad_step = 000452, loss = 0.001517
grad_step = 000453, loss = 0.001516
grad_step = 000454, loss = 0.001518
grad_step = 000455, loss = 0.001522
grad_step = 000456, loss = 0.001527
grad_step = 000457, loss = 0.001536
grad_step = 000458, loss = 0.001554
grad_step = 000459, loss = 0.001578
grad_step = 000460, loss = 0.001625
grad_step = 000461, loss = 0.001663
grad_step = 000462, loss = 0.001726
grad_step = 000463, loss = 0.001697
grad_step = 000464, loss = 0.001646
grad_step = 000465, loss = 0.001566
grad_step = 000466, loss = 0.001544
grad_step = 000467, loss = 0.001557
grad_step = 000468, loss = 0.001543
grad_step = 000469, loss = 0.001536
grad_step = 000470, loss = 0.001554
grad_step = 000471, loss = 0.001563
grad_step = 000472, loss = 0.001545
grad_step = 000473, loss = 0.001505
grad_step = 000474, loss = 0.001496
grad_step = 000475, loss = 0.001520
grad_step = 000476, loss = 0.001531
grad_step = 000477, loss = 0.001516
grad_step = 000478, loss = 0.001494
grad_step = 000479, loss = 0.001493
grad_step = 000480, loss = 0.001507
grad_step = 000481, loss = 0.001509
grad_step = 000482, loss = 0.001496
grad_step = 000483, loss = 0.001482
grad_step = 000484, loss = 0.001481
grad_step = 000485, loss = 0.001488
grad_step = 000486, loss = 0.001489
grad_step = 000487, loss = 0.001482
grad_step = 000488, loss = 0.001474
grad_step = 000489, loss = 0.001473
grad_step = 000490, loss = 0.001478
grad_step = 000491, loss = 0.001481
grad_step = 000492, loss = 0.001480
grad_step = 000493, loss = 0.001485
grad_step = 000494, loss = 0.001502
grad_step = 000495, loss = 0.001539
grad_step = 000496, loss = 0.001580
grad_step = 000497, loss = 0.001652
grad_step = 000498, loss = 0.001662
grad_step = 000499, loss = 0.001667
grad_step = 000500, loss = 0.001580
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001524
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
  (date_run                              2020-05-09 05:04:28.328320
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.203467
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 05:04:28.335784
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 0.0883376
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 05:04:28.345284
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.133355
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 05:04:28.350810
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.342321
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
0  2020-05-09 05:04:09.108907  ...    mean_absolute_error
1  2020-05-09 05:04:09.113023  ...     mean_squared_error
2  2020-05-09 05:04:09.116449  ...  median_absolute_error
3  2020-05-09 05:04:09.119570  ...               r2_score
4  2020-05-09 05:04:28.328320  ...    mean_absolute_error
5  2020-05-09 05:04:28.335784  ...     mean_squared_error
6  2020-05-09 05:04:28.345284  ...  median_absolute_error
7  2020-05-09 05:04:28.350810  ...               r2_score

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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140562679698152
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140561400827800
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140561400377808
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140561400378312
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140561400378816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140561400379320
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fd757370358> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.633780
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.599132
grad_step = 000002, loss = 0.570163
grad_step = 000003, loss = 0.539221
grad_step = 000004, loss = 0.506299
grad_step = 000005, loss = 0.477395
grad_step = 000006, loss = 0.458802
grad_step = 000007, loss = 0.444130
grad_step = 000008, loss = 0.434060
grad_step = 000009, loss = 0.420468
grad_step = 000010, loss = 0.405626
grad_step = 000011, loss = 0.392076
grad_step = 000012, loss = 0.381306
grad_step = 000013, loss = 0.372361
grad_step = 000014, loss = 0.363828
grad_step = 000015, loss = 0.354843
grad_step = 000016, loss = 0.344756
grad_step = 000017, loss = 0.333236
grad_step = 000018, loss = 0.321515
grad_step = 000019, loss = 0.311620
grad_step = 000020, loss = 0.304069
grad_step = 000021, loss = 0.296724
grad_step = 000022, loss = 0.287399
grad_step = 000023, loss = 0.277047
grad_step = 000024, loss = 0.267856
grad_step = 000025, loss = 0.260221
grad_step = 000026, loss = 0.252848
grad_step = 000027, loss = 0.244901
grad_step = 000028, loss = 0.236708
grad_step = 000029, loss = 0.228767
grad_step = 000030, loss = 0.221208
grad_step = 000031, loss = 0.214035
grad_step = 000032, loss = 0.207183
grad_step = 000033, loss = 0.200326
grad_step = 000034, loss = 0.193233
grad_step = 000035, loss = 0.186219
grad_step = 000036, loss = 0.179701
grad_step = 000037, loss = 0.173591
grad_step = 000038, loss = 0.167501
grad_step = 000039, loss = 0.161293
grad_step = 000040, loss = 0.155208
grad_step = 000041, loss = 0.149426
grad_step = 000042, loss = 0.143839
grad_step = 000043, loss = 0.138376
grad_step = 000044, loss = 0.133026
grad_step = 000045, loss = 0.127749
grad_step = 000046, loss = 0.122616
grad_step = 000047, loss = 0.117734
grad_step = 000048, loss = 0.113007
grad_step = 000049, loss = 0.108284
grad_step = 000050, loss = 0.103679
grad_step = 000051, loss = 0.099311
grad_step = 000052, loss = 0.095087
grad_step = 000053, loss = 0.090964
grad_step = 000054, loss = 0.086984
grad_step = 000055, loss = 0.083110
grad_step = 000056, loss = 0.079359
grad_step = 000057, loss = 0.075764
grad_step = 000058, loss = 0.072275
grad_step = 000059, loss = 0.068919
grad_step = 000060, loss = 0.065693
grad_step = 000061, loss = 0.062545
grad_step = 000062, loss = 0.059533
grad_step = 000063, loss = 0.056647
grad_step = 000064, loss = 0.053837
grad_step = 000065, loss = 0.051151
grad_step = 000066, loss = 0.048578
grad_step = 000067, loss = 0.046108
grad_step = 000068, loss = 0.043727
grad_step = 000069, loss = 0.041432
grad_step = 000070, loss = 0.039257
grad_step = 000071, loss = 0.037166
grad_step = 000072, loss = 0.035159
grad_step = 000073, loss = 0.033243
grad_step = 000074, loss = 0.031417
grad_step = 000075, loss = 0.029679
grad_step = 000076, loss = 0.028003
grad_step = 000077, loss = 0.026420
grad_step = 000078, loss = 0.024918
grad_step = 000079, loss = 0.023487
grad_step = 000080, loss = 0.022119
grad_step = 000081, loss = 0.020830
grad_step = 000082, loss = 0.019608
grad_step = 000083, loss = 0.018445
grad_step = 000084, loss = 0.017350
grad_step = 000085, loss = 0.016316
grad_step = 000086, loss = 0.015337
grad_step = 000087, loss = 0.014414
grad_step = 000088, loss = 0.013548
grad_step = 000089, loss = 0.012731
grad_step = 000090, loss = 0.011966
grad_step = 000091, loss = 0.011244
grad_step = 000092, loss = 0.010568
grad_step = 000093, loss = 0.009936
grad_step = 000094, loss = 0.009345
grad_step = 000095, loss = 0.008790
grad_step = 000096, loss = 0.008275
grad_step = 000097, loss = 0.007793
grad_step = 000098, loss = 0.007341
grad_step = 000099, loss = 0.006924
grad_step = 000100, loss = 0.006535
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.006172
grad_step = 000102, loss = 0.005837
grad_step = 000103, loss = 0.005526
grad_step = 000104, loss = 0.005236
grad_step = 000105, loss = 0.004970
grad_step = 000106, loss = 0.004723
grad_step = 000107, loss = 0.004494
grad_step = 000108, loss = 0.004284
grad_step = 000109, loss = 0.004090
grad_step = 000110, loss = 0.003911
grad_step = 000111, loss = 0.003745
grad_step = 000112, loss = 0.003594
grad_step = 000113, loss = 0.003454
grad_step = 000114, loss = 0.003326
grad_step = 000115, loss = 0.003209
grad_step = 000116, loss = 0.003101
grad_step = 000117, loss = 0.003001
grad_step = 000118, loss = 0.002911
grad_step = 000119, loss = 0.002828
grad_step = 000120, loss = 0.002751
grad_step = 000121, loss = 0.002682
grad_step = 000122, loss = 0.002618
grad_step = 000123, loss = 0.002560
grad_step = 000124, loss = 0.002506
grad_step = 000125, loss = 0.002458
grad_step = 000126, loss = 0.002413
grad_step = 000127, loss = 0.002372
grad_step = 000128, loss = 0.002335
grad_step = 000129, loss = 0.002301
grad_step = 000130, loss = 0.002270
grad_step = 000131, loss = 0.002241
grad_step = 000132, loss = 0.002215
grad_step = 000133, loss = 0.002191
grad_step = 000134, loss = 0.002170
grad_step = 000135, loss = 0.002150
grad_step = 000136, loss = 0.002131
grad_step = 000137, loss = 0.002114
grad_step = 000138, loss = 0.002099
grad_step = 000139, loss = 0.002085
grad_step = 000140, loss = 0.002072
grad_step = 000141, loss = 0.002060
grad_step = 000142, loss = 0.002049
grad_step = 000143, loss = 0.002039
grad_step = 000144, loss = 0.002030
grad_step = 000145, loss = 0.002021
grad_step = 000146, loss = 0.002013
grad_step = 000147, loss = 0.002006
grad_step = 000148, loss = 0.002000
grad_step = 000149, loss = 0.001995
grad_step = 000150, loss = 0.001991
grad_step = 000151, loss = 0.001988
grad_step = 000152, loss = 0.001982
grad_step = 000153, loss = 0.001973
grad_step = 000154, loss = 0.001966
grad_step = 000155, loss = 0.001961
grad_step = 000156, loss = 0.001959
grad_step = 000157, loss = 0.001957
grad_step = 000158, loss = 0.001953
grad_step = 000159, loss = 0.001948
grad_step = 000160, loss = 0.001942
grad_step = 000161, loss = 0.001938
grad_step = 000162, loss = 0.001936
grad_step = 000163, loss = 0.001935
grad_step = 000164, loss = 0.001933
grad_step = 000165, loss = 0.001930
grad_step = 000166, loss = 0.001926
grad_step = 000167, loss = 0.001922
grad_step = 000168, loss = 0.001918
grad_step = 000169, loss = 0.001915
grad_step = 000170, loss = 0.001912
grad_step = 000171, loss = 0.001910
grad_step = 000172, loss = 0.001907
grad_step = 000173, loss = 0.001906
grad_step = 000174, loss = 0.001905
grad_step = 000175, loss = 0.001906
grad_step = 000176, loss = 0.001909
grad_step = 000177, loss = 0.001917
grad_step = 000178, loss = 0.001927
grad_step = 000179, loss = 0.001939
grad_step = 000180, loss = 0.001933
grad_step = 000181, loss = 0.001916
grad_step = 000182, loss = 0.001891
grad_step = 000183, loss = 0.001879
grad_step = 000184, loss = 0.001883
grad_step = 000185, loss = 0.001895
grad_step = 000186, loss = 0.001902
grad_step = 000187, loss = 0.001895
grad_step = 000188, loss = 0.001881
grad_step = 000189, loss = 0.001867
grad_step = 000190, loss = 0.001863
grad_step = 000191, loss = 0.001867
grad_step = 000192, loss = 0.001872
grad_step = 000193, loss = 0.001875
grad_step = 000194, loss = 0.001871
grad_step = 000195, loss = 0.001864
grad_step = 000196, loss = 0.001855
grad_step = 000197, loss = 0.001848
grad_step = 000198, loss = 0.001844
grad_step = 000199, loss = 0.001843
grad_step = 000200, loss = 0.001844
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001846
grad_step = 000202, loss = 0.001849
grad_step = 000203, loss = 0.001854
grad_step = 000204, loss = 0.001862
grad_step = 000205, loss = 0.001869
grad_step = 000206, loss = 0.001880
grad_step = 000207, loss = 0.001883
grad_step = 000208, loss = 0.001883
grad_step = 000209, loss = 0.001866
grad_step = 000210, loss = 0.001846
grad_step = 000211, loss = 0.001824
grad_step = 000212, loss = 0.001812
grad_step = 000213, loss = 0.001811
grad_step = 000214, loss = 0.001817
grad_step = 000215, loss = 0.001828
grad_step = 000216, loss = 0.001837
grad_step = 000217, loss = 0.001845
grad_step = 000218, loss = 0.001845
grad_step = 000219, loss = 0.001841
grad_step = 000220, loss = 0.001828
grad_step = 000221, loss = 0.001814
grad_step = 000222, loss = 0.001800
grad_step = 000223, loss = 0.001790
grad_step = 000224, loss = 0.001785
grad_step = 000225, loss = 0.001785
grad_step = 000226, loss = 0.001787
grad_step = 000227, loss = 0.001792
grad_step = 000228, loss = 0.001799
grad_step = 000229, loss = 0.001809
grad_step = 000230, loss = 0.001827
grad_step = 000231, loss = 0.001846
grad_step = 000232, loss = 0.001877
grad_step = 000233, loss = 0.001891
grad_step = 000234, loss = 0.001902
grad_step = 000235, loss = 0.001864
grad_step = 000236, loss = 0.001817
grad_step = 000237, loss = 0.001771
grad_step = 000238, loss = 0.001759
grad_step = 000239, loss = 0.001776
grad_step = 000240, loss = 0.001803
grad_step = 000241, loss = 0.001825
grad_step = 000242, loss = 0.001818
grad_step = 000243, loss = 0.001796
grad_step = 000244, loss = 0.001765
grad_step = 000245, loss = 0.001747
grad_step = 000246, loss = 0.001747
grad_step = 000247, loss = 0.001759
grad_step = 000248, loss = 0.001775
grad_step = 000249, loss = 0.001783
grad_step = 000250, loss = 0.001784
grad_step = 000251, loss = 0.001777
grad_step = 000252, loss = 0.001768
grad_step = 000253, loss = 0.001755
grad_step = 000254, loss = 0.001743
grad_step = 000255, loss = 0.001734
grad_step = 000256, loss = 0.001728
grad_step = 000257, loss = 0.001726
grad_step = 000258, loss = 0.001726
grad_step = 000259, loss = 0.001728
grad_step = 000260, loss = 0.001732
grad_step = 000261, loss = 0.001739
grad_step = 000262, loss = 0.001751
grad_step = 000263, loss = 0.001776
grad_step = 000264, loss = 0.001814
grad_step = 000265, loss = 0.001890
grad_step = 000266, loss = 0.001958
grad_step = 000267, loss = 0.002051
grad_step = 000268, loss = 0.001982
grad_step = 000269, loss = 0.001869
grad_step = 000270, loss = 0.001733
grad_step = 000271, loss = 0.001718
grad_step = 000272, loss = 0.001802
grad_step = 000273, loss = 0.001855
grad_step = 000274, loss = 0.001832
grad_step = 000275, loss = 0.001738
grad_step = 000276, loss = 0.001704
grad_step = 000277, loss = 0.001748
grad_step = 000278, loss = 0.001789
grad_step = 000279, loss = 0.001776
grad_step = 000280, loss = 0.001718
grad_step = 000281, loss = 0.001697
grad_step = 000282, loss = 0.001727
grad_step = 000283, loss = 0.001751
grad_step = 000284, loss = 0.001741
grad_step = 000285, loss = 0.001706
grad_step = 000286, loss = 0.001691
grad_step = 000287, loss = 0.001706
grad_step = 000288, loss = 0.001723
grad_step = 000289, loss = 0.001723
grad_step = 000290, loss = 0.001703
grad_step = 000291, loss = 0.001686
grad_step = 000292, loss = 0.001687
grad_step = 000293, loss = 0.001698
grad_step = 000294, loss = 0.001706
grad_step = 000295, loss = 0.001700
grad_step = 000296, loss = 0.001688
grad_step = 000297, loss = 0.001679
grad_step = 000298, loss = 0.001678
grad_step = 000299, loss = 0.001683
grad_step = 000300, loss = 0.001688
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001689
grad_step = 000302, loss = 0.001685
grad_step = 000303, loss = 0.001678
grad_step = 000304, loss = 0.001672
grad_step = 000305, loss = 0.001669
grad_step = 000306, loss = 0.001668
grad_step = 000307, loss = 0.001669
grad_step = 000308, loss = 0.001671
grad_step = 000309, loss = 0.001672
grad_step = 000310, loss = 0.001674
grad_step = 000311, loss = 0.001676
grad_step = 000312, loss = 0.001678
grad_step = 000313, loss = 0.001680
grad_step = 000314, loss = 0.001684
grad_step = 000315, loss = 0.001688
grad_step = 000316, loss = 0.001697
grad_step = 000317, loss = 0.001707
grad_step = 000318, loss = 0.001727
grad_step = 000319, loss = 0.001745
grad_step = 000320, loss = 0.001780
grad_step = 000321, loss = 0.001799
grad_step = 000322, loss = 0.001832
grad_step = 000323, loss = 0.001816
grad_step = 000324, loss = 0.001797
grad_step = 000325, loss = 0.001733
grad_step = 000326, loss = 0.001681
grad_step = 000327, loss = 0.001649
grad_step = 000328, loss = 0.001650
grad_step = 000329, loss = 0.001675
grad_step = 000330, loss = 0.001700
grad_step = 000331, loss = 0.001717
grad_step = 000332, loss = 0.001703
grad_step = 000333, loss = 0.001681
grad_step = 000334, loss = 0.001653
grad_step = 000335, loss = 0.001639
grad_step = 000336, loss = 0.001641
grad_step = 000337, loss = 0.001654
grad_step = 000338, loss = 0.001667
grad_step = 000339, loss = 0.001669
grad_step = 000340, loss = 0.001664
grad_step = 000341, loss = 0.001651
grad_step = 000342, loss = 0.001639
grad_step = 000343, loss = 0.001632
grad_step = 000344, loss = 0.001630
grad_step = 000345, loss = 0.001633
grad_step = 000346, loss = 0.001638
grad_step = 000347, loss = 0.001643
grad_step = 000348, loss = 0.001645
grad_step = 000349, loss = 0.001646
grad_step = 000350, loss = 0.001644
grad_step = 000351, loss = 0.001640
grad_step = 000352, loss = 0.001635
grad_step = 000353, loss = 0.001630
grad_step = 000354, loss = 0.001625
grad_step = 000355, loss = 0.001621
grad_step = 000356, loss = 0.001619
grad_step = 000357, loss = 0.001617
grad_step = 000358, loss = 0.001615
grad_step = 000359, loss = 0.001614
grad_step = 000360, loss = 0.001614
grad_step = 000361, loss = 0.001613
grad_step = 000362, loss = 0.001613
grad_step = 000363, loss = 0.001614
grad_step = 000364, loss = 0.001616
grad_step = 000365, loss = 0.001620
grad_step = 000366, loss = 0.001628
grad_step = 000367, loss = 0.001642
grad_step = 000368, loss = 0.001673
grad_step = 000369, loss = 0.001722
grad_step = 000370, loss = 0.001827
grad_step = 000371, loss = 0.001937
grad_step = 000372, loss = 0.002131
grad_step = 000373, loss = 0.002091
grad_step = 000374, loss = 0.001992
grad_step = 000375, loss = 0.001702
grad_step = 000376, loss = 0.001604
grad_step = 000377, loss = 0.001726
grad_step = 000378, loss = 0.001803
grad_step = 000379, loss = 0.001735
grad_step = 000380, loss = 0.001610
grad_step = 000381, loss = 0.001651
grad_step = 000382, loss = 0.001739
grad_step = 000383, loss = 0.001674
grad_step = 000384, loss = 0.001602
grad_step = 000385, loss = 0.001638
grad_step = 000386, loss = 0.001678
grad_step = 000387, loss = 0.001642
grad_step = 000388, loss = 0.001595
grad_step = 000389, loss = 0.001619
grad_step = 000390, loss = 0.001654
grad_step = 000391, loss = 0.001628
grad_step = 000392, loss = 0.001592
grad_step = 000393, loss = 0.001594
grad_step = 000394, loss = 0.001618
grad_step = 000395, loss = 0.001622
grad_step = 000396, loss = 0.001596
grad_step = 000397, loss = 0.001581
grad_step = 000398, loss = 0.001592
grad_step = 000399, loss = 0.001605
grad_step = 000400, loss = 0.001599
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001582
grad_step = 000402, loss = 0.001576
grad_step = 000403, loss = 0.001585
grad_step = 000404, loss = 0.001591
grad_step = 000405, loss = 0.001585
grad_step = 000406, loss = 0.001573
grad_step = 000407, loss = 0.001571
grad_step = 000408, loss = 0.001576
grad_step = 000409, loss = 0.001579
grad_step = 000410, loss = 0.001575
grad_step = 000411, loss = 0.001568
grad_step = 000412, loss = 0.001565
grad_step = 000413, loss = 0.001566
grad_step = 000414, loss = 0.001568
grad_step = 000415, loss = 0.001568
grad_step = 000416, loss = 0.001564
grad_step = 000417, loss = 0.001560
grad_step = 000418, loss = 0.001558
grad_step = 000419, loss = 0.001558
grad_step = 000420, loss = 0.001559
grad_step = 000421, loss = 0.001559
grad_step = 000422, loss = 0.001557
grad_step = 000423, loss = 0.001554
grad_step = 000424, loss = 0.001552
grad_step = 000425, loss = 0.001551
grad_step = 000426, loss = 0.001550
grad_step = 000427, loss = 0.001550
grad_step = 000428, loss = 0.001550
grad_step = 000429, loss = 0.001549
grad_step = 000430, loss = 0.001547
grad_step = 000431, loss = 0.001545
grad_step = 000432, loss = 0.001544
grad_step = 000433, loss = 0.001542
grad_step = 000434, loss = 0.001541
grad_step = 000435, loss = 0.001540
grad_step = 000436, loss = 0.001539
grad_step = 000437, loss = 0.001538
grad_step = 000438, loss = 0.001538
grad_step = 000439, loss = 0.001537
grad_step = 000440, loss = 0.001536
grad_step = 000441, loss = 0.001536
grad_step = 000442, loss = 0.001535
grad_step = 000443, loss = 0.001535
grad_step = 000444, loss = 0.001535
grad_step = 000445, loss = 0.001535
grad_step = 000446, loss = 0.001538
grad_step = 000447, loss = 0.001543
grad_step = 000448, loss = 0.001553
grad_step = 000449, loss = 0.001568
grad_step = 000450, loss = 0.001599
grad_step = 000451, loss = 0.001640
grad_step = 000452, loss = 0.001722
grad_step = 000453, loss = 0.001803
grad_step = 000454, loss = 0.001945
grad_step = 000455, loss = 0.001957
grad_step = 000456, loss = 0.001950
grad_step = 000457, loss = 0.001744
grad_step = 000458, loss = 0.001573
grad_step = 000459, loss = 0.001519
grad_step = 000460, loss = 0.001598
grad_step = 000461, loss = 0.001695
grad_step = 000462, loss = 0.001663
grad_step = 000463, loss = 0.001571
grad_step = 000464, loss = 0.001511
grad_step = 000465, loss = 0.001548
grad_step = 000466, loss = 0.001607
grad_step = 000467, loss = 0.001586
grad_step = 000468, loss = 0.001528
grad_step = 000469, loss = 0.001508
grad_step = 000470, loss = 0.001541
grad_step = 000471, loss = 0.001567
grad_step = 000472, loss = 0.001540
grad_step = 000473, loss = 0.001506
grad_step = 000474, loss = 0.001507
grad_step = 000475, loss = 0.001530
grad_step = 000476, loss = 0.001535
grad_step = 000477, loss = 0.001512
grad_step = 000478, loss = 0.001496
grad_step = 000479, loss = 0.001502
grad_step = 000480, loss = 0.001515
grad_step = 000481, loss = 0.001514
grad_step = 000482, loss = 0.001501
grad_step = 000483, loss = 0.001491
grad_step = 000484, loss = 0.001491
grad_step = 000485, loss = 0.001498
grad_step = 000486, loss = 0.001502
grad_step = 000487, loss = 0.001498
grad_step = 000488, loss = 0.001491
grad_step = 000489, loss = 0.001485
grad_step = 000490, loss = 0.001484
grad_step = 000491, loss = 0.001486
grad_step = 000492, loss = 0.001485
grad_step = 000493, loss = 0.001484
grad_step = 000494, loss = 0.001484
grad_step = 000495, loss = 0.001484
grad_step = 000496, loss = 0.001484
grad_step = 000497, loss = 0.001481
grad_step = 000498, loss = 0.001477
grad_step = 000499, loss = 0.001474
grad_step = 000500, loss = 0.001473
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001473
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
  (date_run                              2020-05-09 05:04:51.091957
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.24052
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 05:04:51.097567
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.13915
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 05:04:51.105331
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.138651
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 05:04:51.110292
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.11443
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fd7573703c8> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355306.2500
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 221598.7344
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 112055.7422
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 50766.7930
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 25411.7598
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 14645.6699
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 9453.4238
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 6655.7524
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 5021.9077
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 4013.4497
  ('#### Inference Need return ypred, ytrue #########################',) 
[[ 3.24723125e-01 -1.05850840e+00 -2.22686410e+00 -6.90285563e-01
  -6.56494617e-01  8.93949211e-01  8.11673164e-01 -1.36328653e-01
  -2.11064410e+00  5.12161374e-01  2.02825069e+00  3.48047256e-01
   4.10854012e-01  7.90889621e-01  3.99612308e-01  1.13566732e+00
   8.68562818e-01  3.06718200e-01 -2.02326846e+00  8.17117751e-01
  -4.16574508e-01 -1.57839417e+00 -1.59624267e+00 -1.58774495e-01
   7.35869229e-01 -3.00060242e-01 -2.55601192e+00  2.81349182e-01
   1.83905697e+00 -5.78744411e-02 -2.59395981e+00 -7.05032408e-01
   9.80939746e-01 -2.64451003e+00  1.08637798e+00 -2.01119852e+00
   1.03769445e+00 -8.53465199e-02 -2.32726932e-02 -2.27008772e+00
   1.13123333e+00 -3.60282958e-01  2.53083920e+00 -1.25104570e+00
   3.59428704e-01  1.62995720e+00  1.67294812e+00  7.63896942e-01
  -9.53226924e-01  6.09233141e-01  1.28939211e+00  1.09456718e-01
  -1.71711540e+00  1.06106782e+00 -1.45635355e+00 -2.73796177e+00
   1.16915715e+00  5.35800755e-02  7.59094179e-01 -3.78869593e-01
   4.82074380e-01  9.39990807e+00  1.09661350e+01  1.09781027e+01
   9.76259518e+00  1.17477865e+01  1.16999369e+01  1.06638508e+01
   1.10838699e+01  1.05449638e+01  7.88582468e+00  9.90381241e+00
   1.10972748e+01  1.00218306e+01  1.09390440e+01  1.16617727e+01
   1.01770048e+01  1.20943384e+01  1.26245394e+01  1.07481508e+01
   1.19334841e+01  1.17127008e+01  1.10510416e+01  1.04362774e+01
   1.07794037e+01  1.15236721e+01  1.15123510e+01  1.11578083e+01
   1.13566904e+01  1.03148479e+01  9.53607273e+00  1.03033304e+01
   1.38755760e+01  1.04654188e+01  8.48917198e+00  1.19832973e+01
   1.13236628e+01  1.35374985e+01  1.09572763e+01  1.10029974e+01
   9.16756344e+00  1.12116928e+01  1.27287388e+01  1.22963104e+01
   1.14608173e+01  1.25832186e+01  1.00704737e+01  1.15339098e+01
   1.28948793e+01  1.15691528e+01  9.55557728e+00  9.93640804e+00
   1.30053902e+01  1.15930948e+01  1.15747919e+01  1.09966736e+01
   9.49701118e+00  1.01023579e+01  9.52083206e+00  9.99866772e+00
  -2.39609957e+00 -7.06166029e-01  6.86240196e-03 -1.73953402e+00
  -4.17291760e-01 -2.10208249e+00 -2.02105331e+00 -1.84096861e+00
   1.33438796e-01  7.11891651e-02  3.76854897e-01  2.15237665e+00
  -5.82576394e-01 -2.18881989e+00  1.50024760e+00  6.82151020e-02
   1.02329159e+00  1.78470850e+00 -1.64060533e-01 -1.22409940e-01
  -2.50645965e-01  4.75702047e-01 -1.44725847e+00 -1.61009634e+00
   2.57037616e+00  1.03834558e+00 -9.82807875e-01  1.49849391e+00
  -8.28988314e-01  2.97730386e-01 -2.39553213e-01  1.91528350e-01
  -1.38970041e+00  1.41889632e-01 -1.64462996e+00 -5.97715378e-01
  -9.04582977e-01  2.23579600e-01 -2.33851981e+00 -6.55209005e-01
  -2.30860621e-01  2.91383266e-02  3.30387139e+00 -3.97186995e-01
   5.48660815e-01 -1.15323997e+00 -1.75346506e+00 -1.75555515e+00
  -1.16822034e-01  2.37298369e+00 -9.69692647e-01  1.03591055e-01
  -1.12609315e+00 -1.58160627e+00 -3.33642960e+00  1.10643172e+00
   6.81027353e-01  2.06440997e+00  1.44534111e+00  1.39054441e+00
   2.00963306e+00  9.37358141e-02  1.14056194e+00  6.15697920e-01
   5.98856807e-02  1.84931159e-01  1.03493071e+00  1.71203065e+00
   7.93033123e-01  1.14068222e+00  1.69033194e+00  2.17142916e+00
   1.45511889e+00  2.44029641e-01  1.33475125e-01  1.25703168e+00
   8.55686367e-01  1.37068534e+00  3.90690804e-01  7.94409692e-01
   2.16519642e+00  1.85775280e-01  6.94161594e-01  1.14573979e+00
   1.33693147e+00  4.59833503e-01  5.45758247e-01  5.42225063e-01
   3.25080752e-02  7.87439108e-01  1.06366885e+00  5.08764982e-02
   4.12131250e-01  3.28126848e-01  1.47718179e+00  1.11287940e+00
   5.59366882e-01  2.71507263e-01  9.74985719e-01  3.97182345e-01
   9.92211759e-01  2.24951553e+00  3.21538782e+00  2.15197861e-01
   6.95029914e-01  1.24873996e-01  3.01775575e-01  1.72519040e+00
   1.85991859e+00  5.44230342e-01  4.99134719e-01  2.31887579e-01
   9.97397304e-02  3.38637829e-01  1.19188488e-01  2.94967532e-01
   2.42754316e+00  2.06076479e+00  2.96839237e-01  2.21542895e-01
   4.78972793e-01  1.09211788e+01  1.10904675e+01  1.11295462e+01
   1.18454342e+01  8.68644238e+00  1.20934649e+01  8.64930344e+00
   1.06012554e+01  1.15017176e+01  9.98215961e+00  1.28087397e+01
   1.37159634e+01  1.03373766e+01  1.11859169e+01  1.10658112e+01
   1.33394623e+01  1.24940767e+01  1.34948263e+01  1.22312326e+01
   1.19533482e+01  1.29934072e+01  1.22078724e+01  1.04518003e+01
   9.79713631e+00  9.18447590e+00  1.21302223e+01  1.21617880e+01
   1.09322748e+01  1.05540380e+01  1.19552670e+01  9.62564373e+00
   1.02762423e+01  1.05380087e+01  9.62080288e+00  1.02655144e+01
   1.25248299e+01  1.09893141e+01  1.23857155e+01  9.20622826e+00
   1.05784464e+01  8.39472580e+00  1.15645256e+01  1.09147091e+01
   1.07127457e+01  9.68009377e+00  1.19754601e+01  1.19407520e+01
   1.12174931e+01  1.19854422e+01  1.12609568e+01  8.65086842e+00
   8.95196438e+00  1.22488184e+01  1.26382780e+01  1.04870577e+01
   1.32900724e+01  1.12925587e+01  1.12421875e+01  1.03177691e+01
   2.21668673e+00  1.26756835e+00  8.99212956e-02  2.88494766e-01
   1.58291984e+00  1.18084240e+00  1.89822137e-01  3.25437164e+00
   2.07133532e+00  1.77861714e+00  5.37051857e-01  3.27280402e-01
   7.13175535e-02  2.36197376e+00  1.30267024e-01  5.97237945e-01
   4.29791629e-01  1.63496518e+00  2.30832767e+00  6.13933861e-01
   6.47242486e-01  3.45081282e+00  1.61789322e+00  1.50171626e+00
   7.61414051e-01  2.59056211e-01  1.54984713e+00  2.46056795e+00
   1.59629738e+00  8.89202952e-02  1.34195292e+00  3.55392873e-01
   1.45008135e+00  1.22903526e-01  4.86275434e-01  4.00776815e+00
   1.59339833e+00  1.66087878e+00  1.92043984e+00  1.14048302e-01
   2.37304306e+00  6.60384178e-01  1.83095694e-01  3.35293388e+00
   5.25923967e-01  4.32332695e-01  1.84568429e+00  3.38070869e-01
   9.94462371e-02  2.07232714e+00  3.85032082e+00  9.44474399e-01
   2.54580975e-01  6.29241168e-01  7.71133244e-01  2.20929980e+00
   1.48561084e+00  6.84468091e-01  6.54848933e-01  9.25846159e-01
  -1.34344034e+01  5.86498260e+00 -1.00192423e+01]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 05:04:59.790371
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   91.9488
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 05:04:59.794137
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    8478.9
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 05:04:59.798051
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   92.3484
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 05:04:59.801185
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -758.345
metric_name                                             r2_score
Name: 7, dtype: object,) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon/fb_prophet.py'},) 
  ('#### Setup Model   ##############################################',) 
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fd750fcd8d0> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  ({'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, FileNotFoundError(2, "File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist")) 
  ("### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, KeyError('model_uri',)) 
  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/',) 
  (                     date_run  ...            metric_name
0  2020-05-09 05:04:51.091957  ...    mean_absolute_error
1  2020-05-09 05:04:51.097567  ...     mean_squared_error
2  2020-05-09 05:04:51.105331  ...  median_absolute_error
3  2020-05-09 05:04:51.110292  ...               r2_score
4  2020-05-09 05:04:59.790371  ...    mean_absolute_error
5  2020-05-09 05:04:59.794137  ...     mean_squared_error
6  2020-05-09 05:04:59.798051  ...  median_absolute_error
7  2020-05-09 05:04:59.801185  ...               r2_score

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

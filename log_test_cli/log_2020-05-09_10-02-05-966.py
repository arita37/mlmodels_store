  ('/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json',) 
  ('test_cli', 'GITHUB_REPOSITORT', 'GITHUB_SHA') 
  ('Running command', 'test_cli') 
  ('# Testing Command Line System  ',) 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d53e2c42d4ebc720e72bd423c3f46fa090c6a954', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'd53e2c42d4ebc720e72bd423c3f46fa090c6a954', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d53e2c42d4ebc720e72bd423c3f46fa090c6a954

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d53e2c42d4ebc720e72bd423c3f46fa090c6a954

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
  (<mlmodels.model_tf.1_lstm.Model object at 0x7fed805507f0>,) 
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
 [ 0.0768115  -0.00660114 -0.08954524  0.01756427  0.02690193  0.1004031 ]
 [ 0.02757698 -0.0545092   0.05201496  0.04842753  0.37787431  0.14264864]
 [ 0.22548911  0.0248813  -0.02743954  0.14962421  0.12835532  0.08546653]
 [ 0.21660614  0.22267407 -0.13267523  0.34085593  0.34180182  0.11488639]
 [ 0.38745865 -0.09118826  0.06117924  0.59876424  0.21139666 -0.24120545]
 [-0.01140661 -0.14302072 -0.15689372  0.03867501  0.07107762  0.26337022]
 [-0.20911318 -0.06313886  0.07242882  0.16184883  0.05893778 -0.03811092]
 [ 0.15041855 -0.15926032  0.15226921  0.41678944  0.14395864 -0.32517168]
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
{'loss': 0.46951039135456085, 'loss_history': []}
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
{'loss': 0.5921013504266739, 'loss_history': []}
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
[32m[I 2020-05-09 10:02:34,452][0m Finished trial#0 resulted in value: 2.2854484021663666. Current best value is 2.2854484021663666 with parameters: {'learning_rate': 0.021395787506325443, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-09 10:02:36,375][0m Finished trial#1 resulted in value: 10.673916816711426. Current best value is 2.2854484021663666 with parameters: {'learning_rate': 0.021395787506325443, 'num_layers': 3, 'size': 6, 'output_size': 6, 'size_layer': 256, 'timestep': 5, 'epoch': 2}.[0m
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f2e4d0c5390> <class 'mlmodels.model_gluon.fb_prophet.Model'>
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2e423c2128> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355423.4688
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 272692.5938
Epoch 3/10

1/1 [==============================] - 0s 106ms/step - loss: 181866.2188
Epoch 4/10

1/1 [==============================] - 0s 108ms/step - loss: 111467.5078
Epoch 5/10

1/1 [==============================] - 0s 152ms/step - loss: 66064.1953
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 39881.4219
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 25409.8594
Epoch 8/10

1/1 [==============================] - 0s 107ms/step - loss: 17169.5762
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 12288.7598
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 9166.2344
  ('#### Inference Need return ypred, ytrue #########################',) 
[[-2.9019439e-01  8.8285315e-01 -7.7319568e-01  2.3575255e-01
   1.3467791e+00 -6.6647971e-01 -5.2504236e-01 -6.7367578e-01
  -1.9070292e-01  6.8283254e-01 -4.4199073e-01  7.0218152e-01
   7.6434100e-01 -2.4865015e-01 -5.1494670e-01 -1.6120169e-01
  -1.8352777e-01 -4.1392979e-01  1.6677804e+00  9.5611483e-01
   8.4796929e-01  1.3054535e+00 -8.1081712e-01 -9.7497761e-01
   1.0557091e+00  3.1523407e-01  1.2540045e+00 -1.3701664e+00
   1.2024491e+00  7.3463124e-01 -5.6176621e-01  9.3513399e-01
   1.9499916e-01  1.3595064e+00  1.0692887e+00 -6.2279862e-01
   6.4186794e-01  3.7233913e-01 -1.0771115e+00  3.6145949e-01
   8.5645401e-01 -4.1226834e-02  5.9127605e-01 -6.8973994e-01
  -2.3478472e+00  6.9657910e-01 -3.8817444e-01 -1.0057143e+00
   3.2269204e-01  3.6553514e-01 -1.4743304e+00  2.4987680e-01
   9.1941786e-01  8.1665970e-02 -1.2951651e+00 -6.4843792e-01
  -7.6687372e-01  1.9425600e+00 -1.0986620e+00  2.1461602e-01
  -2.8635773e-01  9.2735994e-01 -6.8801332e-01  3.5759383e-01
  -3.9704868e-01 -2.9691929e-01 -1.5808784e+00  1.7399008e+00
  -1.0141323e+00  9.7449684e-01 -2.9053566e-01  2.0971537e-02
   5.9323937e-01  7.8271979e-01 -9.9599719e-01  1.7415370e-01
  -1.2509338e+00  7.6550299e-01  3.4705365e-01  4.3127549e-01
  -1.0049175e+00 -1.3060356e+00 -4.1825280e-01  9.1259730e-01
   7.5240785e-01  1.8159014e+00 -6.1720246e-01  1.1500125e+00
   1.3272102e+00  5.8414733e-01 -3.0689605e-02 -5.4762363e-03
  -4.2863451e-02 -3.4916773e-01 -1.6224738e+00 -4.0193999e-01
   3.3284903e-02  1.4086357e+00  3.4093595e-01 -6.2056112e-01
   1.1921716e+00  4.1524869e-01  4.8710549e-01  7.2511220e-01
  -3.4971279e-01  3.8403955e-01  1.6537452e+00 -8.5220194e-01
  -1.5461074e+00 -1.1444639e+00  2.3951200e-01  5.9080708e-01
  -1.6833148e+00  4.1509154e-01 -1.2976797e+00  3.7693739e-02
   2.2359763e-01 -1.6467139e+00  1.0658671e+00 -1.5639788e+00
   2.7149540e-01  8.1238041e+00  5.2075067e+00  7.4826832e+00
   6.7689691e+00  7.5766282e+00  5.4636712e+00  6.2792025e+00
   6.7656851e+00  6.1098351e+00  7.2448092e+00  7.3282866e+00
   6.1649179e+00  8.0398512e+00  7.0689549e+00  5.8419380e+00
   6.2360740e+00  7.6821165e+00  5.5534267e+00  7.4837489e+00
   6.4201760e+00  7.1390619e+00  7.3300753e+00  7.4006600e+00
   6.0612698e+00  7.1228251e+00  5.9586453e+00  5.9787803e+00
   4.8533659e+00  7.0140028e+00  7.4316726e+00  6.9935546e+00
   6.5714417e+00  6.0752616e+00  5.5914884e+00  6.6934423e+00
   8.0726252e+00  6.5299206e+00  5.6900330e+00  7.3347464e+00
   6.5507183e+00  6.6033506e+00  6.3631234e+00  7.0743546e+00
   6.2967644e+00  5.0242510e+00  6.9843664e+00  6.8857460e+00
   7.0397921e+00  7.1903796e+00  8.1510649e+00  7.3615150e+00
   7.8600903e+00  5.8819294e+00  5.5376053e+00  7.0910482e+00
   6.3815408e+00  8.0323277e+00  7.3867545e+00  5.3909235e+00
   8.2534331e-01  9.3424475e-01  6.5742594e-01  6.3382268e-01
   5.2397978e-01  1.5286627e+00  1.5597241e+00  4.2549491e-01
   1.6095841e+00  1.0242990e+00  1.7677354e+00  1.7921112e+00
   1.7547917e-01  1.7756221e+00  2.2128409e-01  6.8736780e-01
   3.1045735e-01  9.6046603e-01  9.2365754e-01  2.1175241e+00
   1.4573972e+00  1.1261631e+00  1.0234702e+00  1.5818621e+00
   1.6836286e+00  1.4173638e+00  2.8343064e-01  8.5170269e-01
   5.6815249e-01  1.9553831e+00  3.2805526e-01  2.0202651e+00
   1.6276250e+00  5.0516748e-01  1.4946471e+00  3.5638529e-01
   9.5793355e-01  1.0466892e+00  2.7888513e-01  1.4106826e+00
   7.8675914e-01  3.7225103e-01  1.4409764e+00  1.7955310e+00
   6.1222160e-01  1.1110290e+00  1.3027000e-01  1.3650362e+00
   1.5906051e+00  1.7074594e+00  1.7998326e+00  2.2415316e+00
   1.0642736e+00  2.1762948e+00  7.9670227e-01  2.3123388e+00
   6.9221592e-01  1.0539918e+00  2.7008975e-01  3.4132123e-01
   1.4183681e+00  1.6262525e+00  5.8599555e-01  2.4018760e+00
   1.7434261e+00  1.7123209e+00  1.1452569e+00  3.1225407e-01
   9.6874976e-01  1.4265527e+00  6.6289115e-01  4.8117411e-01
   9.0967053e-01  1.3949373e+00  2.5209150e+00  2.9278760e+00
   5.1170546e-01  1.4082530e+00  3.6257207e-01  1.9713148e+00
   1.6413126e+00  2.1731007e-01  1.7648869e+00  3.4968895e-01
   7.3410672e-01  1.2400181e+00  1.6952682e+00  7.1898025e-01
   4.3896163e-01  1.2415537e+00  1.5395532e+00  4.6886504e-01
   2.1735184e+00  2.5482559e-01  5.9272683e-01  5.0397295e-01
   8.7364656e-01  3.0819786e-01  1.5603569e+00  1.5582025e-01
   1.2569438e+00  6.3119066e-01  4.6214455e-01  1.9780934e-01
   6.4933771e-01  4.7572482e-01  2.0635614e+00  1.0484121e+00
   1.2963921e+00  2.3772135e+00  4.1272521e-01  5.8093584e-01
   1.6494596e+00  3.5582036e-01  5.7795012e-01  6.6939938e-01
   2.0989001e-01  6.0705352e-01  1.9976115e+00  4.8248374e-01
   3.0200422e-02  6.8176303e+00  5.4154439e+00  6.9365606e+00
   6.3324366e+00  6.7262053e+00  6.2152328e+00  7.2283783e+00
   7.4910545e+00  6.3034697e+00  6.1966000e+00  7.7440724e+00
   7.1849856e+00  6.7328601e+00  5.9724264e+00  6.8554969e+00
   6.2813330e+00  7.0147309e+00  7.6293283e+00  6.5656800e+00
   6.0592670e+00  8.1607981e+00  7.3385005e+00  8.2128210e+00
   6.0004754e+00  8.0062046e+00  8.4436073e+00  6.2295489e+00
   8.7383842e+00  7.2982564e+00  7.3305326e+00  6.3652682e+00
   8.0904779e+00  8.0400276e+00  8.8537083e+00  6.3483772e+00
   8.8665295e+00  8.2895489e+00  6.1169338e+00  6.6104989e+00
   7.1894102e+00  7.9671898e+00  7.2405300e+00  7.9850979e+00
   8.3493919e+00  7.4321346e+00  6.8948379e+00  6.7338300e+00
   7.6036510e+00  6.8839335e+00  7.5483809e+00  7.8314862e+00
   6.3512626e+00  7.0001245e+00  6.3339701e+00  8.1025877e+00
   7.4971294e+00  6.6081586e+00  7.5999613e+00  7.6129460e+00
  -1.2655951e+00 -6.1557193e+00  3.2969816e+00]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 10:02:49.180243
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.4096
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 10:02:49.184788
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   9122.04
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 10:02:49.188377
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.2104
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 10:02:49.192005
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -815.943
metric_name                                             r2_score
Name: 3, dtype: object,) 
  ("### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256},) 
  ('#### Setup Model   ##############################################',) 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139836322384360
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139835315809528
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139835315810032
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139835315421480
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139835315421984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139835315422488
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2e4d0c5400> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.422143
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.396751
grad_step = 000002, loss = 0.373365
grad_step = 000003, loss = 0.347566
grad_step = 000004, loss = 0.319861
grad_step = 000005, loss = 0.293712
grad_step = 000006, loss = 0.277030
grad_step = 000007, loss = 0.274696
grad_step = 000008, loss = 0.263382
grad_step = 000009, loss = 0.245466
grad_step = 000010, loss = 0.231535
grad_step = 000011, loss = 0.223124
grad_step = 000012, loss = 0.216575
grad_step = 000013, loss = 0.209067
grad_step = 000014, loss = 0.199829
grad_step = 000015, loss = 0.189397
grad_step = 000016, loss = 0.179016
grad_step = 000017, loss = 0.170306
grad_step = 000018, loss = 0.163360
grad_step = 000019, loss = 0.155675
grad_step = 000020, loss = 0.146218
grad_step = 000021, loss = 0.137081
grad_step = 000022, loss = 0.129741
grad_step = 000023, loss = 0.123615
grad_step = 000024, loss = 0.117541
grad_step = 000025, loss = 0.110991
grad_step = 000026, loss = 0.104123
grad_step = 000027, loss = 0.097505
grad_step = 000028, loss = 0.091595
grad_step = 000029, loss = 0.086328
grad_step = 000030, loss = 0.081185
grad_step = 000031, loss = 0.075858
grad_step = 000032, loss = 0.070747
grad_step = 000033, loss = 0.066249
grad_step = 000034, loss = 0.062134
grad_step = 000035, loss = 0.058031
grad_step = 000036, loss = 0.053962
grad_step = 000037, loss = 0.050257
grad_step = 000038, loss = 0.047061
grad_step = 000039, loss = 0.044068
grad_step = 000040, loss = 0.041001
grad_step = 000041, loss = 0.038003
grad_step = 000042, loss = 0.035366
grad_step = 000043, loss = 0.033074
grad_step = 000044, loss = 0.030860
grad_step = 000045, loss = 0.028626
grad_step = 000046, loss = 0.026533
grad_step = 000047, loss = 0.024709
grad_step = 000048, loss = 0.023050
grad_step = 000049, loss = 0.021543
grad_step = 000050, loss = 0.019974
grad_step = 000051, loss = 0.018595
grad_step = 000052, loss = 0.017341
grad_step = 000053, loss = 0.016172
grad_step = 000054, loss = 0.015075
grad_step = 000055, loss = 0.014025
grad_step = 000056, loss = 0.013085
grad_step = 000057, loss = 0.012250
grad_step = 000058, loss = 0.011424
grad_step = 000059, loss = 0.010660
grad_step = 000060, loss = 0.009987
grad_step = 000061, loss = 0.009353
grad_step = 000062, loss = 0.008762
grad_step = 000063, loss = 0.008207
grad_step = 000064, loss = 0.007690
grad_step = 000065, loss = 0.007240
grad_step = 000066, loss = 0.006807
grad_step = 000067, loss = 0.006392
grad_step = 000068, loss = 0.006030
grad_step = 000069, loss = 0.005691
grad_step = 000070, loss = 0.005378
grad_step = 000071, loss = 0.005092
grad_step = 000072, loss = 0.004827
grad_step = 000073, loss = 0.004588
grad_step = 000074, loss = 0.004360
grad_step = 000075, loss = 0.004150
grad_step = 000076, loss = 0.003963
grad_step = 000077, loss = 0.003791
grad_step = 000078, loss = 0.003632
grad_step = 000079, loss = 0.003485
grad_step = 000080, loss = 0.003352
grad_step = 000081, loss = 0.003235
grad_step = 000082, loss = 0.003123
grad_step = 000083, loss = 0.003021
grad_step = 000084, loss = 0.002930
grad_step = 000085, loss = 0.002846
grad_step = 000086, loss = 0.002771
grad_step = 000087, loss = 0.002701
grad_step = 000088, loss = 0.002640
grad_step = 000089, loss = 0.002584
grad_step = 000090, loss = 0.002533
grad_step = 000091, loss = 0.002487
grad_step = 000092, loss = 0.002447
grad_step = 000093, loss = 0.002411
grad_step = 000094, loss = 0.002377
grad_step = 000095, loss = 0.002348
grad_step = 000096, loss = 0.002323
grad_step = 000097, loss = 0.002300
grad_step = 000098, loss = 0.002280
grad_step = 000099, loss = 0.002262
grad_step = 000100, loss = 0.002248
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002234
grad_step = 000102, loss = 0.002223
grad_step = 000103, loss = 0.002213
grad_step = 000104, loss = 0.002204
grad_step = 000105, loss = 0.002196
grad_step = 000106, loss = 0.002190
grad_step = 000107, loss = 0.002184
grad_step = 000108, loss = 0.002179
grad_step = 000109, loss = 0.002174
grad_step = 000110, loss = 0.002169
grad_step = 000111, loss = 0.002165
grad_step = 000112, loss = 0.002161
grad_step = 000113, loss = 0.002157
grad_step = 000114, loss = 0.002154
grad_step = 000115, loss = 0.002150
grad_step = 000116, loss = 0.002146
grad_step = 000117, loss = 0.002143
grad_step = 000118, loss = 0.002139
grad_step = 000119, loss = 0.002135
grad_step = 000120, loss = 0.002132
grad_step = 000121, loss = 0.002128
grad_step = 000122, loss = 0.002124
grad_step = 000123, loss = 0.002120
grad_step = 000124, loss = 0.002117
grad_step = 000125, loss = 0.002113
grad_step = 000126, loss = 0.002109
grad_step = 000127, loss = 0.002105
grad_step = 000128, loss = 0.002101
grad_step = 000129, loss = 0.002097
grad_step = 000130, loss = 0.002093
grad_step = 000131, loss = 0.002089
grad_step = 000132, loss = 0.002085
grad_step = 000133, loss = 0.002081
grad_step = 000134, loss = 0.002076
grad_step = 000135, loss = 0.002072
grad_step = 000136, loss = 0.002068
grad_step = 000137, loss = 0.002064
grad_step = 000138, loss = 0.002059
grad_step = 000139, loss = 0.002054
grad_step = 000140, loss = 0.002050
grad_step = 000141, loss = 0.002046
grad_step = 000142, loss = 0.002041
grad_step = 000143, loss = 0.002035
grad_step = 000144, loss = 0.002030
grad_step = 000145, loss = 0.002025
grad_step = 000146, loss = 0.002019
grad_step = 000147, loss = 0.002014
grad_step = 000148, loss = 0.002010
grad_step = 000149, loss = 0.002007
grad_step = 000150, loss = 0.002003
grad_step = 000151, loss = 0.001996
grad_step = 000152, loss = 0.001987
grad_step = 000153, loss = 0.001979
grad_step = 000154, loss = 0.001973
grad_step = 000155, loss = 0.001969
grad_step = 000156, loss = 0.001967
grad_step = 000157, loss = 0.001967
grad_step = 000158, loss = 0.001970
grad_step = 000159, loss = 0.001966
grad_step = 000160, loss = 0.001956
grad_step = 000161, loss = 0.001941
grad_step = 000162, loss = 0.001926
grad_step = 000163, loss = 0.001918
grad_step = 000164, loss = 0.001917
grad_step = 000165, loss = 0.001920
grad_step = 000166, loss = 0.001932
grad_step = 000167, loss = 0.001946
grad_step = 000168, loss = 0.001957
grad_step = 000169, loss = 0.001942
grad_step = 000170, loss = 0.001917
grad_step = 000171, loss = 0.001885
grad_step = 000172, loss = 0.001869
grad_step = 000173, loss = 0.001868
grad_step = 000174, loss = 0.001867
grad_step = 000175, loss = 0.001873
grad_step = 000176, loss = 0.001898
grad_step = 000177, loss = 0.001927
grad_step = 000178, loss = 0.001931
grad_step = 000179, loss = 0.001933
grad_step = 000180, loss = 0.001897
grad_step = 000181, loss = 0.001840
grad_step = 000182, loss = 0.001825
grad_step = 000183, loss = 0.001855
grad_step = 000184, loss = 0.001890
grad_step = 000185, loss = 0.001891
grad_step = 000186, loss = 0.001870
grad_step = 000187, loss = 0.001820
grad_step = 000188, loss = 0.001810
grad_step = 000189, loss = 0.001832
grad_step = 000190, loss = 0.001836
grad_step = 000191, loss = 0.001828
grad_step = 000192, loss = 0.001807
grad_step = 000193, loss = 0.001786
grad_step = 000194, loss = 0.001782
grad_step = 000195, loss = 0.001788
grad_step = 000196, loss = 0.001786
grad_step = 000197, loss = 0.001780
grad_step = 000198, loss = 0.001789
grad_step = 000199, loss = 0.001830
grad_step = 000200, loss = 0.001900
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002025
grad_step = 000202, loss = 0.002071
grad_step = 000203, loss = 0.002049
grad_step = 000204, loss = 0.001889
grad_step = 000205, loss = 0.001773
grad_step = 000206, loss = 0.001910
grad_step = 000207, loss = 0.001952
grad_step = 000208, loss = 0.001788
grad_step = 000209, loss = 0.001807
grad_step = 000210, loss = 0.001880
grad_step = 000211, loss = 0.001841
grad_step = 000212, loss = 0.001796
grad_step = 000213, loss = 0.001785
grad_step = 000214, loss = 0.001806
grad_step = 000215, loss = 0.001821
grad_step = 000216, loss = 0.001749
grad_step = 000217, loss = 0.001754
grad_step = 000218, loss = 0.001791
grad_step = 000219, loss = 0.001761
grad_step = 000220, loss = 0.001727
grad_step = 000221, loss = 0.001759
grad_step = 000222, loss = 0.001750
grad_step = 000223, loss = 0.001733
grad_step = 000224, loss = 0.001737
grad_step = 000225, loss = 0.001727
grad_step = 000226, loss = 0.001724
grad_step = 000227, loss = 0.001730
grad_step = 000228, loss = 0.001721
grad_step = 000229, loss = 0.001703
grad_step = 000230, loss = 0.001715
grad_step = 000231, loss = 0.001719
grad_step = 000232, loss = 0.001703
grad_step = 000233, loss = 0.001697
grad_step = 000234, loss = 0.001703
grad_step = 000235, loss = 0.001698
grad_step = 000236, loss = 0.001691
grad_step = 000237, loss = 0.001694
grad_step = 000238, loss = 0.001691
grad_step = 000239, loss = 0.001684
grad_step = 000240, loss = 0.001681
grad_step = 000241, loss = 0.001683
grad_step = 000242, loss = 0.001682
grad_step = 000243, loss = 0.001676
grad_step = 000244, loss = 0.001674
grad_step = 000245, loss = 0.001673
grad_step = 000246, loss = 0.001670
grad_step = 000247, loss = 0.001665
grad_step = 000248, loss = 0.001663
grad_step = 000249, loss = 0.001663
grad_step = 000250, loss = 0.001661
grad_step = 000251, loss = 0.001658
grad_step = 000252, loss = 0.001655
grad_step = 000253, loss = 0.001655
grad_step = 000254, loss = 0.001654
grad_step = 000255, loss = 0.001652
grad_step = 000256, loss = 0.001651
grad_step = 000257, loss = 0.001652
grad_step = 000258, loss = 0.001657
grad_step = 000259, loss = 0.001666
grad_step = 000260, loss = 0.001685
grad_step = 000261, loss = 0.001716
grad_step = 000262, loss = 0.001758
grad_step = 000263, loss = 0.001809
grad_step = 000264, loss = 0.001803
grad_step = 000265, loss = 0.001746
grad_step = 000266, loss = 0.001672
grad_step = 000267, loss = 0.001625
grad_step = 000268, loss = 0.001634
grad_step = 000269, loss = 0.001684
grad_step = 000270, loss = 0.001725
grad_step = 000271, loss = 0.001732
grad_step = 000272, loss = 0.001681
grad_step = 000273, loss = 0.001624
grad_step = 000274, loss = 0.001603
grad_step = 000275, loss = 0.001622
grad_step = 000276, loss = 0.001647
grad_step = 000277, loss = 0.001644
grad_step = 000278, loss = 0.001616
grad_step = 000279, loss = 0.001591
grad_step = 000280, loss = 0.001588
grad_step = 000281, loss = 0.001602
grad_step = 000282, loss = 0.001615
grad_step = 000283, loss = 0.001617
grad_step = 000284, loss = 0.001605
grad_step = 000285, loss = 0.001595
grad_step = 000286, loss = 0.001596
grad_step = 000287, loss = 0.001618
grad_step = 000288, loss = 0.001631
grad_step = 000289, loss = 0.001625
grad_step = 000290, loss = 0.001608
grad_step = 000291, loss = 0.001601
grad_step = 000292, loss = 0.001607
grad_step = 000293, loss = 0.001578
grad_step = 000294, loss = 0.001549
grad_step = 000295, loss = 0.001541
grad_step = 000296, loss = 0.001560
grad_step = 000297, loss = 0.001581
grad_step = 000298, loss = 0.001563
grad_step = 000299, loss = 0.001546
grad_step = 000300, loss = 0.001542
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001542
grad_step = 000302, loss = 0.001532
grad_step = 000303, loss = 0.001513
grad_step = 000304, loss = 0.001503
grad_step = 000305, loss = 0.001508
grad_step = 000306, loss = 0.001517
grad_step = 000307, loss = 0.001522
grad_step = 000308, loss = 0.001525
grad_step = 000309, loss = 0.001550
grad_step = 000310, loss = 0.001616
grad_step = 000311, loss = 0.001707
grad_step = 000312, loss = 0.001840
grad_step = 000313, loss = 0.001804
grad_step = 000314, loss = 0.001698
grad_step = 000315, loss = 0.001589
grad_step = 000316, loss = 0.001531
grad_step = 000317, loss = 0.001537
grad_step = 000318, loss = 0.001555
grad_step = 000319, loss = 0.001601
grad_step = 000320, loss = 0.001563
grad_step = 000321, loss = 0.001453
grad_step = 000322, loss = 0.001484
grad_step = 000323, loss = 0.001554
grad_step = 000324, loss = 0.001503
grad_step = 000325, loss = 0.001453
grad_step = 000326, loss = 0.001501
grad_step = 000327, loss = 0.001478
grad_step = 000328, loss = 0.001417
grad_step = 000329, loss = 0.001434
grad_step = 000330, loss = 0.001458
grad_step = 000331, loss = 0.001438
grad_step = 000332, loss = 0.001410
grad_step = 000333, loss = 0.001439
grad_step = 000334, loss = 0.001448
grad_step = 000335, loss = 0.001412
grad_step = 000336, loss = 0.001429
grad_step = 000337, loss = 0.001452
grad_step = 000338, loss = 0.001443
grad_step = 000339, loss = 0.001463
grad_step = 000340, loss = 0.001515
grad_step = 000341, loss = 0.001574
grad_step = 000342, loss = 0.001634
grad_step = 000343, loss = 0.001705
grad_step = 000344, loss = 0.001676
grad_step = 000345, loss = 0.001524
grad_step = 000346, loss = 0.001396
grad_step = 000347, loss = 0.001364
grad_step = 000348, loss = 0.001430
grad_step = 000349, loss = 0.001492
grad_step = 000350, loss = 0.001453
grad_step = 000351, loss = 0.001375
grad_step = 000352, loss = 0.001340
grad_step = 000353, loss = 0.001369
grad_step = 000354, loss = 0.001415
grad_step = 000355, loss = 0.001401
grad_step = 000356, loss = 0.001355
grad_step = 000357, loss = 0.001326
grad_step = 000358, loss = 0.001328
grad_step = 000359, loss = 0.001349
grad_step = 000360, loss = 0.001359
grad_step = 000361, loss = 0.001335
grad_step = 000362, loss = 0.001305
grad_step = 000363, loss = 0.001290
grad_step = 000364, loss = 0.001296
grad_step = 000365, loss = 0.001318
grad_step = 000366, loss = 0.001336
grad_step = 000367, loss = 0.001346
grad_step = 000368, loss = 0.001337
grad_step = 000369, loss = 0.001315
grad_step = 000370, loss = 0.001283
grad_step = 000371, loss = 0.001268
grad_step = 000372, loss = 0.001273
grad_step = 000373, loss = 0.001286
grad_step = 000374, loss = 0.001288
grad_step = 000375, loss = 0.001263
grad_step = 000376, loss = 0.001236
grad_step = 000377, loss = 0.001220
grad_step = 000378, loss = 0.001215
grad_step = 000379, loss = 0.001220
grad_step = 000380, loss = 0.001237
grad_step = 000381, loss = 0.001257
grad_step = 000382, loss = 0.001274
grad_step = 000383, loss = 0.001289
grad_step = 000384, loss = 0.001263
grad_step = 000385, loss = 0.001221
grad_step = 000386, loss = 0.001180
grad_step = 000387, loss = 0.001166
grad_step = 000388, loss = 0.001178
grad_step = 000389, loss = 0.001198
grad_step = 000390, loss = 0.001194
grad_step = 000391, loss = 0.001171
grad_step = 000392, loss = 0.001165
grad_step = 000393, loss = 0.001185
grad_step = 000394, loss = 0.001189
grad_step = 000395, loss = 0.001175
grad_step = 000396, loss = 0.001161
grad_step = 000397, loss = 0.001191
grad_step = 000398, loss = 0.001226
grad_step = 000399, loss = 0.001227
grad_step = 000400, loss = 0.001220
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001188
grad_step = 000402, loss = 0.001175
grad_step = 000403, loss = 0.001168
grad_step = 000404, loss = 0.001118
grad_step = 000405, loss = 0.001081
grad_step = 000406, loss = 0.001085
grad_step = 000407, loss = 0.001059
grad_step = 000408, loss = 0.001036
grad_step = 000409, loss = 0.001026
grad_step = 000410, loss = 0.001024
grad_step = 000411, loss = 0.001036
grad_step = 000412, loss = 0.001065
grad_step = 000413, loss = 0.001069
grad_step = 000414, loss = 0.001079
grad_step = 000415, loss = 0.001133
grad_step = 000416, loss = 0.001178
grad_step = 000417, loss = 0.001202
grad_step = 000418, loss = 0.001250
grad_step = 000419, loss = 0.001250
grad_step = 000420, loss = 0.001031
grad_step = 000421, loss = 0.000992
grad_step = 000422, loss = 0.001089
grad_step = 000423, loss = 0.001020
grad_step = 000424, loss = 0.001109
grad_step = 000425, loss = 0.001195
grad_step = 000426, loss = 0.000944
grad_step = 000427, loss = 0.001055
grad_step = 000428, loss = 0.001031
grad_step = 000429, loss = 0.000981
grad_step = 000430, loss = 0.001057
grad_step = 000431, loss = 0.000894
grad_step = 000432, loss = 0.000948
grad_step = 000433, loss = 0.000919
grad_step = 000434, loss = 0.000856
grad_step = 000435, loss = 0.000915
grad_step = 000436, loss = 0.000900
grad_step = 000437, loss = 0.000908
grad_step = 000438, loss = 0.000972
grad_step = 000439, loss = 0.000962
grad_step = 000440, loss = 0.000971
grad_step = 000441, loss = 0.000966
grad_step = 000442, loss = 0.000902
grad_step = 000443, loss = 0.000881
grad_step = 000444, loss = 0.000797
grad_step = 000445, loss = 0.000793
grad_step = 000446, loss = 0.000854
grad_step = 000447, loss = 0.000865
grad_step = 000448, loss = 0.000893
grad_step = 000449, loss = 0.000840
grad_step = 000450, loss = 0.000780
grad_step = 000451, loss = 0.000758
grad_step = 000452, loss = 0.000726
grad_step = 000453, loss = 0.000718
grad_step = 000454, loss = 0.000746
grad_step = 000455, loss = 0.000743
grad_step = 000456, loss = 0.000724
grad_step = 000457, loss = 0.000717
grad_step = 000458, loss = 0.000685
grad_step = 000459, loss = 0.000673
grad_step = 000460, loss = 0.000674
grad_step = 000461, loss = 0.000677
grad_step = 000462, loss = 0.000679
grad_step = 000463, loss = 0.000698
grad_step = 000464, loss = 0.000716
grad_step = 000465, loss = 0.000736
grad_step = 000466, loss = 0.000755
grad_step = 000467, loss = 0.000770
grad_step = 000468, loss = 0.000760
grad_step = 000469, loss = 0.000724
grad_step = 000470, loss = 0.000663
grad_step = 000471, loss = 0.000616
grad_step = 000472, loss = 0.000602
grad_step = 000473, loss = 0.000619
grad_step = 000474, loss = 0.000651
grad_step = 000475, loss = 0.000658
grad_step = 000476, loss = 0.000640
grad_step = 000477, loss = 0.000605
grad_step = 000478, loss = 0.000576
grad_step = 000479, loss = 0.000561
grad_step = 000480, loss = 0.000565
grad_step = 000481, loss = 0.000584
grad_step = 000482, loss = 0.000600
grad_step = 000483, loss = 0.000619
grad_step = 000484, loss = 0.000631
grad_step = 000485, loss = 0.000628
grad_step = 000486, loss = 0.000601
grad_step = 000487, loss = 0.000569
grad_step = 000488, loss = 0.000537
grad_step = 000489, loss = 0.000519
grad_step = 000490, loss = 0.000516
grad_step = 000491, loss = 0.000523
grad_step = 000492, loss = 0.000532
grad_step = 000493, loss = 0.000536
grad_step = 000494, loss = 0.000539
grad_step = 000495, loss = 0.000533
grad_step = 000496, loss = 0.000523
grad_step = 000497, loss = 0.000511
grad_step = 000498, loss = 0.000501
grad_step = 000499, loss = 0.000491
grad_step = 000500, loss = 0.000483
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000477
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
  (date_run                              2020-05-09 10:03:11.747295
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.27159
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 10:03:11.753702
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   0.20509
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 10:03:11.759623
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.140149
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 10:03:11.765350
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -2.11642
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
0  2020-05-09 10:02:49.180243  ...    mean_absolute_error
1  2020-05-09 10:02:49.184788  ...     mean_squared_error
2  2020-05-09 10:02:49.188377  ...  median_absolute_error
3  2020-05-09 10:02:49.192005  ...               r2_score
4  2020-05-09 10:03:11.747295  ...    mean_absolute_error
5  2020-05-09 10:03:11.753702  ...     mean_squared_error
6  2020-05-09 10:03:11.759623  ...  median_absolute_error
7  2020-05-09 10:03:11.765350  ...               r2_score

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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140631430111976
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140630151237472
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140630150787480
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140630150787984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140630150788488
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140630150788992
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fe7590ef390> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.595855
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.562530
grad_step = 000002, loss = 0.533857
grad_step = 000003, loss = 0.504256
grad_step = 000004, loss = 0.473033
grad_step = 000005, loss = 0.444718
grad_step = 000006, loss = 0.425073
grad_step = 000007, loss = 0.407464
grad_step = 000008, loss = 0.397387
grad_step = 000009, loss = 0.386384
grad_step = 000010, loss = 0.369997
grad_step = 000011, loss = 0.356027
grad_step = 000012, loss = 0.347659
grad_step = 000013, loss = 0.339589
grad_step = 000014, loss = 0.328515
grad_step = 000015, loss = 0.316357
grad_step = 000016, loss = 0.304614
grad_step = 000017, loss = 0.293602
grad_step = 000018, loss = 0.283431
grad_step = 000019, loss = 0.274309
grad_step = 000020, loss = 0.265090
grad_step = 000021, loss = 0.254820
grad_step = 000022, loss = 0.244442
grad_step = 000023, loss = 0.235200
grad_step = 000024, loss = 0.226783
grad_step = 000025, loss = 0.218496
grad_step = 000026, loss = 0.210305
grad_step = 000027, loss = 0.202170
grad_step = 000028, loss = 0.193911
grad_step = 000029, loss = 0.185713
grad_step = 000030, loss = 0.178086
grad_step = 000031, loss = 0.171185
grad_step = 000032, loss = 0.164328
grad_step = 000033, loss = 0.157114
grad_step = 000034, loss = 0.150092
grad_step = 000035, loss = 0.143668
grad_step = 000036, loss = 0.137625
grad_step = 000037, loss = 0.131621
grad_step = 000038, loss = 0.125575
grad_step = 000039, loss = 0.119693
grad_step = 000040, loss = 0.114277
grad_step = 000041, loss = 0.109181
grad_step = 000042, loss = 0.103962
grad_step = 000043, loss = 0.098789
grad_step = 000044, loss = 0.094031
grad_step = 000045, loss = 0.089537
grad_step = 000046, loss = 0.085077
grad_step = 000047, loss = 0.080727
grad_step = 000048, loss = 0.076589
grad_step = 000049, loss = 0.072664
grad_step = 000050, loss = 0.068919
grad_step = 000051, loss = 0.065296
grad_step = 000052, loss = 0.061789
grad_step = 000053, loss = 0.058457
grad_step = 000054, loss = 0.055302
grad_step = 000055, loss = 0.052265
grad_step = 000056, loss = 0.049329
grad_step = 000057, loss = 0.046557
grad_step = 000058, loss = 0.043962
grad_step = 000059, loss = 0.041439
grad_step = 000060, loss = 0.039007
grad_step = 000061, loss = 0.036768
grad_step = 000062, loss = 0.034633
grad_step = 000063, loss = 0.032544
grad_step = 000064, loss = 0.030588
grad_step = 000065, loss = 0.028759
grad_step = 000066, loss = 0.027006
grad_step = 000067, loss = 0.025328
grad_step = 000068, loss = 0.023759
grad_step = 000069, loss = 0.022280
grad_step = 000070, loss = 0.020870
grad_step = 000071, loss = 0.019537
grad_step = 000072, loss = 0.018286
grad_step = 000073, loss = 0.017116
grad_step = 000074, loss = 0.015989
grad_step = 000075, loss = 0.014943
grad_step = 000076, loss = 0.013970
grad_step = 000077, loss = 0.013042
grad_step = 000078, loss = 0.012178
grad_step = 000079, loss = 0.011380
grad_step = 000080, loss = 0.010622
grad_step = 000081, loss = 0.009917
grad_step = 000082, loss = 0.009267
grad_step = 000083, loss = 0.008655
grad_step = 000084, loss = 0.008090
grad_step = 000085, loss = 0.007570
grad_step = 000086, loss = 0.007087
grad_step = 000087, loss = 0.006638
grad_step = 000088, loss = 0.006227
grad_step = 000089, loss = 0.005848
grad_step = 000090, loss = 0.005496
grad_step = 000091, loss = 0.005176
grad_step = 000092, loss = 0.004883
grad_step = 000093, loss = 0.004611
grad_step = 000094, loss = 0.004366
grad_step = 000095, loss = 0.004139
grad_step = 000096, loss = 0.003932
grad_step = 000097, loss = 0.003744
grad_step = 000098, loss = 0.003572
grad_step = 000099, loss = 0.003415
grad_step = 000100, loss = 0.003273
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003142
grad_step = 000102, loss = 0.003024
grad_step = 000103, loss = 0.002917
grad_step = 000104, loss = 0.002819
grad_step = 000105, loss = 0.002731
grad_step = 000106, loss = 0.002656
grad_step = 000107, loss = 0.002617
grad_step = 000108, loss = 0.002631
grad_step = 000109, loss = 0.002626
grad_step = 000110, loss = 0.002420
grad_step = 000111, loss = 0.002425
grad_step = 000112, loss = 0.002458
grad_step = 000113, loss = 0.002287
grad_step = 000114, loss = 0.002348
grad_step = 000115, loss = 0.002316
grad_step = 000116, loss = 0.002211
grad_step = 000117, loss = 0.002292
grad_step = 000118, loss = 0.002191
grad_step = 000119, loss = 0.002185
grad_step = 000120, loss = 0.002195
grad_step = 000121, loss = 0.002122
grad_step = 000122, loss = 0.002156
grad_step = 000123, loss = 0.002111
grad_step = 000124, loss = 0.002108
grad_step = 000125, loss = 0.002110
grad_step = 000126, loss = 0.002078
grad_step = 000127, loss = 0.002094
grad_step = 000128, loss = 0.002073
grad_step = 000129, loss = 0.002068
grad_step = 000130, loss = 0.002069
grad_step = 000131, loss = 0.002054
grad_step = 000132, loss = 0.002053
grad_step = 000133, loss = 0.002045
grad_step = 000134, loss = 0.002045
grad_step = 000135, loss = 0.002035
grad_step = 000136, loss = 0.002025
grad_step = 000137, loss = 0.002035
grad_step = 000138, loss = 0.002021
grad_step = 000139, loss = 0.002008
grad_step = 000140, loss = 0.002016
grad_step = 000141, loss = 0.002010
grad_step = 000142, loss = 0.002007
grad_step = 000143, loss = 0.002004
grad_step = 000144, loss = 0.001992
grad_step = 000145, loss = 0.001988
grad_step = 000146, loss = 0.001985
grad_step = 000147, loss = 0.001978
grad_step = 000148, loss = 0.001978
grad_step = 000149, loss = 0.001978
grad_step = 000150, loss = 0.001985
grad_step = 000151, loss = 0.002037
grad_step = 000152, loss = 0.002186
grad_step = 000153, loss = 0.002480
grad_step = 000154, loss = 0.002137
grad_step = 000155, loss = 0.002012
grad_step = 000156, loss = 0.002192
grad_step = 000157, loss = 0.002031
grad_step = 000158, loss = 0.002102
grad_step = 000159, loss = 0.002063
grad_step = 000160, loss = 0.002042
grad_step = 000161, loss = 0.002088
grad_step = 000162, loss = 0.001976
grad_step = 000163, loss = 0.002068
grad_step = 000164, loss = 0.001986
grad_step = 000165, loss = 0.001983
grad_step = 000166, loss = 0.002029
grad_step = 000167, loss = 0.001938
grad_step = 000168, loss = 0.001998
grad_step = 000169, loss = 0.001970
grad_step = 000170, loss = 0.001938
grad_step = 000171, loss = 0.001987
grad_step = 000172, loss = 0.001932
grad_step = 000173, loss = 0.001949
grad_step = 000174, loss = 0.001958
grad_step = 000175, loss = 0.001918
grad_step = 000176, loss = 0.001949
grad_step = 000177, loss = 0.001930
grad_step = 000178, loss = 0.001915
grad_step = 000179, loss = 0.001937
grad_step = 000180, loss = 0.001911
grad_step = 000181, loss = 0.001913
grad_step = 000182, loss = 0.001922
grad_step = 000183, loss = 0.001900
grad_step = 000184, loss = 0.001908
grad_step = 000185, loss = 0.001907
grad_step = 000186, loss = 0.001892
grad_step = 000187, loss = 0.001900
grad_step = 000188, loss = 0.001896
grad_step = 000189, loss = 0.001886
grad_step = 000190, loss = 0.001892
grad_step = 000191, loss = 0.001886
grad_step = 000192, loss = 0.001879
grad_step = 000193, loss = 0.001883
grad_step = 000194, loss = 0.001878
grad_step = 000195, loss = 0.001872
grad_step = 000196, loss = 0.001875
grad_step = 000197, loss = 0.001871
grad_step = 000198, loss = 0.001865
grad_step = 000199, loss = 0.001866
grad_step = 000200, loss = 0.001864
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001858
grad_step = 000202, loss = 0.001858
grad_step = 000203, loss = 0.001856
grad_step = 000204, loss = 0.001852
grad_step = 000205, loss = 0.001850
grad_step = 000206, loss = 0.001849
grad_step = 000207, loss = 0.001846
grad_step = 000208, loss = 0.001842
grad_step = 000209, loss = 0.001841
grad_step = 000210, loss = 0.001840
grad_step = 000211, loss = 0.001837
grad_step = 000212, loss = 0.001834
grad_step = 000213, loss = 0.001834
grad_step = 000214, loss = 0.001835
grad_step = 000215, loss = 0.001840
grad_step = 000216, loss = 0.001852
grad_step = 000217, loss = 0.001885
grad_step = 000218, loss = 0.001949
grad_step = 000219, loss = 0.002066
grad_step = 000220, loss = 0.002097
grad_step = 000221, loss = 0.002047
grad_step = 000222, loss = 0.001874
grad_step = 000223, loss = 0.001830
grad_step = 000224, loss = 0.001911
grad_step = 000225, loss = 0.001942
grad_step = 000226, loss = 0.001884
grad_step = 000227, loss = 0.001816
grad_step = 000228, loss = 0.001858
grad_step = 000229, loss = 0.001909
grad_step = 000230, loss = 0.001845
grad_step = 000231, loss = 0.001803
grad_step = 000232, loss = 0.001850
grad_step = 000233, loss = 0.001860
grad_step = 000234, loss = 0.001818
grad_step = 000235, loss = 0.001803
grad_step = 000236, loss = 0.001825
grad_step = 000237, loss = 0.001834
grad_step = 000238, loss = 0.001813
grad_step = 000239, loss = 0.001794
grad_step = 000240, loss = 0.001802
grad_step = 000241, loss = 0.001818
grad_step = 000242, loss = 0.001807
grad_step = 000243, loss = 0.001785
grad_step = 000244, loss = 0.001787
grad_step = 000245, loss = 0.001801
grad_step = 000246, loss = 0.001796
grad_step = 000247, loss = 0.001783
grad_step = 000248, loss = 0.001778
grad_step = 000249, loss = 0.001780
grad_step = 000250, loss = 0.001784
grad_step = 000251, loss = 0.001782
grad_step = 000252, loss = 0.001773
grad_step = 000253, loss = 0.001767
grad_step = 000254, loss = 0.001770
grad_step = 000255, loss = 0.001773
grad_step = 000256, loss = 0.001770
grad_step = 000257, loss = 0.001764
grad_step = 000258, loss = 0.001760
grad_step = 000259, loss = 0.001758
grad_step = 000260, loss = 0.001758
grad_step = 000261, loss = 0.001759
grad_step = 000262, loss = 0.001759
grad_step = 000263, loss = 0.001756
grad_step = 000264, loss = 0.001752
grad_step = 000265, loss = 0.001749
grad_step = 000266, loss = 0.001747
grad_step = 000267, loss = 0.001745
grad_step = 000268, loss = 0.001744
grad_step = 000269, loss = 0.001744
grad_step = 000270, loss = 0.001744
grad_step = 000271, loss = 0.001744
grad_step = 000272, loss = 0.001744
grad_step = 000273, loss = 0.001745
grad_step = 000274, loss = 0.001748
grad_step = 000275, loss = 0.001754
grad_step = 000276, loss = 0.001763
grad_step = 000277, loss = 0.001782
grad_step = 000278, loss = 0.001810
grad_step = 000279, loss = 0.001858
grad_step = 000280, loss = 0.001908
grad_step = 000281, loss = 0.001961
grad_step = 000282, loss = 0.001940
grad_step = 000283, loss = 0.001868
grad_step = 000284, loss = 0.001760
grad_step = 000285, loss = 0.001721
grad_step = 000286, loss = 0.001764
grad_step = 000287, loss = 0.001816
grad_step = 000288, loss = 0.001817
grad_step = 000289, loss = 0.001759
grad_step = 000290, loss = 0.001717
grad_step = 000291, loss = 0.001727
grad_step = 000292, loss = 0.001762
grad_step = 000293, loss = 0.001773
grad_step = 000294, loss = 0.001745
grad_step = 000295, loss = 0.001713
grad_step = 000296, loss = 0.001707
grad_step = 000297, loss = 0.001726
grad_step = 000298, loss = 0.001742
grad_step = 000299, loss = 0.001734
grad_step = 000300, loss = 0.001714
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001700
grad_step = 000302, loss = 0.001700
grad_step = 000303, loss = 0.001709
grad_step = 000304, loss = 0.001715
grad_step = 000305, loss = 0.001713
grad_step = 000306, loss = 0.001703
grad_step = 000307, loss = 0.001693
grad_step = 000308, loss = 0.001688
grad_step = 000309, loss = 0.001691
grad_step = 000310, loss = 0.001696
grad_step = 000311, loss = 0.001698
grad_step = 000312, loss = 0.001695
grad_step = 000313, loss = 0.001689
grad_step = 000314, loss = 0.001683
grad_step = 000315, loss = 0.001679
grad_step = 000316, loss = 0.001677
grad_step = 000317, loss = 0.001677
grad_step = 000318, loss = 0.001678
grad_step = 000319, loss = 0.001680
grad_step = 000320, loss = 0.001681
grad_step = 000321, loss = 0.001682
grad_step = 000322, loss = 0.001683
grad_step = 000323, loss = 0.001684
grad_step = 000324, loss = 0.001685
grad_step = 000325, loss = 0.001689
grad_step = 000326, loss = 0.001693
grad_step = 000327, loss = 0.001701
grad_step = 000328, loss = 0.001711
grad_step = 000329, loss = 0.001727
grad_step = 000330, loss = 0.001745
grad_step = 000331, loss = 0.001768
grad_step = 000332, loss = 0.001783
grad_step = 000333, loss = 0.001789
grad_step = 000334, loss = 0.001769
grad_step = 000335, loss = 0.001733
grad_step = 000336, loss = 0.001687
grad_step = 000337, loss = 0.001658
grad_step = 000338, loss = 0.001651
grad_step = 000339, loss = 0.001662
grad_step = 000340, loss = 0.001680
grad_step = 000341, loss = 0.001692
grad_step = 000342, loss = 0.001695
grad_step = 000343, loss = 0.001683
grad_step = 000344, loss = 0.001667
grad_step = 000345, loss = 0.001649
grad_step = 000346, loss = 0.001638
grad_step = 000347, loss = 0.001637
grad_step = 000348, loss = 0.001644
grad_step = 000349, loss = 0.001653
grad_step = 000350, loss = 0.001657
grad_step = 000351, loss = 0.001658
grad_step = 000352, loss = 0.001653
grad_step = 000353, loss = 0.001648
grad_step = 000354, loss = 0.001641
grad_step = 000355, loss = 0.001635
grad_step = 000356, loss = 0.001630
grad_step = 000357, loss = 0.001626
grad_step = 000358, loss = 0.001623
grad_step = 000359, loss = 0.001620
grad_step = 000360, loss = 0.001618
grad_step = 000361, loss = 0.001617
grad_step = 000362, loss = 0.001617
grad_step = 000363, loss = 0.001618
grad_step = 000364, loss = 0.001621
grad_step = 000365, loss = 0.001626
grad_step = 000366, loss = 0.001635
grad_step = 000367, loss = 0.001653
grad_step = 000368, loss = 0.001685
grad_step = 000369, loss = 0.001740
grad_step = 000370, loss = 0.001830
grad_step = 000371, loss = 0.001938
grad_step = 000372, loss = 0.002065
grad_step = 000373, loss = 0.001999
grad_step = 000374, loss = 0.001835
grad_step = 000375, loss = 0.001635
grad_step = 000376, loss = 0.001637
grad_step = 000377, loss = 0.001766
grad_step = 000378, loss = 0.001782
grad_step = 000379, loss = 0.001688
grad_step = 000380, loss = 0.001618
grad_step = 000381, loss = 0.001644
grad_step = 000382, loss = 0.001706
grad_step = 000383, loss = 0.001694
grad_step = 000384, loss = 0.001621
grad_step = 000385, loss = 0.001594
grad_step = 000386, loss = 0.001641
grad_step = 000387, loss = 0.001667
grad_step = 000388, loss = 0.001630
grad_step = 000389, loss = 0.001597
grad_step = 000390, loss = 0.001599
grad_step = 000391, loss = 0.001608
grad_step = 000392, loss = 0.001611
grad_step = 000393, loss = 0.001612
grad_step = 000394, loss = 0.001604
grad_step = 000395, loss = 0.001584
grad_step = 000396, loss = 0.001579
grad_step = 000397, loss = 0.001590
grad_step = 000398, loss = 0.001592
grad_step = 000399, loss = 0.001581
grad_step = 000400, loss = 0.001575
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001577
grad_step = 000402, loss = 0.001574
grad_step = 000403, loss = 0.001568
grad_step = 000404, loss = 0.001569
grad_step = 000405, loss = 0.001571
grad_step = 000406, loss = 0.001566
grad_step = 000407, loss = 0.001558
grad_step = 000408, loss = 0.001557
grad_step = 000409, loss = 0.001559
grad_step = 000410, loss = 0.001558
grad_step = 000411, loss = 0.001554
grad_step = 000412, loss = 0.001552
grad_step = 000413, loss = 0.001552
grad_step = 000414, loss = 0.001551
grad_step = 000415, loss = 0.001546
grad_step = 000416, loss = 0.001542
grad_step = 000417, loss = 0.001541
grad_step = 000418, loss = 0.001541
grad_step = 000419, loss = 0.001539
grad_step = 000420, loss = 0.001537
grad_step = 000421, loss = 0.001536
grad_step = 000422, loss = 0.001535
grad_step = 000423, loss = 0.001534
grad_step = 000424, loss = 0.001531
grad_step = 000425, loss = 0.001528
grad_step = 000426, loss = 0.001527
grad_step = 000427, loss = 0.001525
grad_step = 000428, loss = 0.001524
grad_step = 000429, loss = 0.001522
grad_step = 000430, loss = 0.001520
grad_step = 000431, loss = 0.001518
grad_step = 000432, loss = 0.001518
grad_step = 000433, loss = 0.001517
grad_step = 000434, loss = 0.001517
grad_step = 000435, loss = 0.001518
grad_step = 000436, loss = 0.001523
grad_step = 000437, loss = 0.001532
grad_step = 000438, loss = 0.001552
grad_step = 000439, loss = 0.001586
grad_step = 000440, loss = 0.001655
grad_step = 000441, loss = 0.001763
grad_step = 000442, loss = 0.001942
grad_step = 000443, loss = 0.002132
grad_step = 000444, loss = 0.002144
grad_step = 000445, loss = 0.001877
grad_step = 000446, loss = 0.001550
grad_step = 000447, loss = 0.001590
grad_step = 000448, loss = 0.001783
grad_step = 000449, loss = 0.001780
grad_step = 000450, loss = 0.001641
grad_step = 000451, loss = 0.001613
grad_step = 000452, loss = 0.001579
grad_step = 000453, loss = 0.001641
grad_step = 000454, loss = 0.001675
grad_step = 000455, loss = 0.001521
grad_step = 000456, loss = 0.001583
grad_step = 000457, loss = 0.001630
grad_step = 000458, loss = 0.001532
grad_step = 000459, loss = 0.001523
grad_step = 000460, loss = 0.001563
grad_step = 000461, loss = 0.001540
grad_step = 000462, loss = 0.001533
grad_step = 000463, loss = 0.001519
grad_step = 000464, loss = 0.001490
grad_step = 000465, loss = 0.001540
grad_step = 000466, loss = 0.001511
grad_step = 000467, loss = 0.001467
grad_step = 000468, loss = 0.001507
grad_step = 000469, loss = 0.001503
grad_step = 000470, loss = 0.001476
grad_step = 000471, loss = 0.001475
grad_step = 000472, loss = 0.001478
grad_step = 000473, loss = 0.001463
grad_step = 000474, loss = 0.001471
grad_step = 000475, loss = 0.001472
grad_step = 000476, loss = 0.001448
grad_step = 000477, loss = 0.001454
grad_step = 000478, loss = 0.001461
grad_step = 000479, loss = 0.001446
grad_step = 000480, loss = 0.001436
grad_step = 000481, loss = 0.001442
grad_step = 000482, loss = 0.001438
grad_step = 000483, loss = 0.001431
grad_step = 000484, loss = 0.001432
grad_step = 000485, loss = 0.001427
grad_step = 000486, loss = 0.001422
grad_step = 000487, loss = 0.001421
grad_step = 000488, loss = 0.001422
grad_step = 000489, loss = 0.001415
grad_step = 000490, loss = 0.001408
grad_step = 000491, loss = 0.001410
grad_step = 000492, loss = 0.001409
grad_step = 000493, loss = 0.001403
grad_step = 000494, loss = 0.001400
grad_step = 000495, loss = 0.001398
grad_step = 000496, loss = 0.001397
grad_step = 000497, loss = 0.001394
grad_step = 000498, loss = 0.001390
grad_step = 000499, loss = 0.001390
grad_step = 000500, loss = 0.001389
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001389
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
  (date_run                              2020-05-09 10:03:37.569820
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.232883
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 10:03:37.574797
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.148291
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 10:03:37.580970
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.124436
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 10:03:37.587082
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.25334
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fe7590ef400> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 355944.2812
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 282806.5938
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 181992.4219
Epoch 4/10

1/1 [==============================] - 0s 99ms/step - loss: 111655.1875
Epoch 5/10

1/1 [==============================] - 0s 103ms/step - loss: 66859.0625
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 41457.7969
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 27137.0137
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 18693.5898
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 13493.8506
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 10157.3115
  ('#### Inference Need return ypred, ytrue #########################',) 
[[-2.50873208e-01  5.41954935e-01 -1.15877315e-01 -9.60155010e-01
  -5.18847883e-01  1.06067196e-01  1.29385948e+00 -7.31892169e-01
   1.47074938e-01 -3.47639471e-01  1.05231628e-01 -2.59479225e-01
  -3.64596903e-01  3.48737538e-02 -2.59124517e-01 -3.85183513e-01
  -6.65488899e-01  5.26223779e-01  1.11399567e+00  3.68490756e-01
  -2.35268071e-01  3.49519640e-01 -1.48296967e-01  1.51043952e+00
   6.53853774e-01  8.07763457e-01  9.80767608e-01  5.82890153e-01
  -5.96192122e-01  2.85803229e-01  1.84768066e-01 -1.58128500e-01
   5.30712962e-01 -7.88379312e-02  4.61731821e-01  3.32027078e-02
   1.39858484e-01  4.93432969e-01 -1.11888218e+00 -3.90854478e-03
   1.84314466e+00 -1.15415907e+00 -4.65286344e-01  1.88353553e-01
   5.76293349e-01  1.17891893e-01  2.56858885e-01 -1.19313455e+00
  -4.75908101e-01  1.03225231e-01 -2.14966917e+00 -5.22527769e-02
  -5.81576586e-01 -1.06357348e+00  2.43582934e-01  6.10709012e-01
  -1.44262064e+00 -2.35166267e-01 -4.86427724e-01  2.92462856e-01
   2.00768277e-01  7.02907085e+00  6.02988291e+00  7.10751200e+00
   6.46229839e+00  6.77365971e+00  5.71530008e+00  5.37594414e+00
   5.39488268e+00  6.47914982e+00  7.03026962e+00  6.53551149e+00
   5.54232168e+00  7.22652626e+00  7.46207380e+00  7.22078800e+00
   5.30765820e+00  5.53009844e+00  6.04258490e+00  6.08870602e+00
   6.51838350e+00  7.01838684e+00  5.24228191e+00  7.27722359e+00
   6.02661228e+00  4.91283560e+00  6.54271507e+00  5.64351702e+00
   6.48421955e+00  5.65191936e+00  5.82731009e+00  7.36057854e+00
   5.55174637e+00  5.95578146e+00  7.14144993e+00  7.05715561e+00
   5.60091066e+00  6.12447786e+00  6.16261673e+00  6.88254881e+00
   4.80053616e+00  6.72045708e+00  5.39927340e+00  6.04599380e+00
   6.95884848e+00  5.47118807e+00  5.88884354e+00  7.76749134e+00
   6.12610912e+00  7.14853859e+00  6.35272837e+00  6.55999565e+00
   5.19951868e+00  7.69690180e+00  6.61521912e+00  6.01156855e+00
   6.93486166e+00  5.45579433e+00  7.14528513e+00  6.00498819e+00
   3.82343590e-01 -2.11031184e-01 -3.27426851e-01  2.63999999e-01
   6.44655347e-01  3.87637287e-01  3.44033241e-01 -1.03837654e-01
  -6.41818225e-01 -2.11761922e-01  1.24978864e+00  2.96415687e-01
   7.79713869e-01 -5.12423038e-01  2.11755916e-01 -3.41200978e-02
   4.42074478e-01  1.94406480e-01  8.23479891e-01 -1.66285503e+00
   6.47390842e-01  9.71799135e-01 -1.11027694e+00 -3.30246449e-01
   6.11131430e-01  4.28202331e-01 -3.31338465e-01  1.14698559e-01
   5.63974828e-02  3.29618156e-01  4.59950298e-01  1.30987573e+00
   8.01340759e-01  5.87336600e-01 -3.00728679e-01 -3.96614224e-02
   9.96808648e-01 -1.17484272e-01 -6.34267211e-01  6.04127169e-01
  -1.08790958e+00 -9.26572740e-01 -6.51786089e-01 -2.59914309e-01
  -5.43916047e-01 -2.76262432e-01  3.73062462e-01 -4.19220686e-01
  -1.94857776e-01  7.28581995e-02  2.12612972e-01  8.06945145e-01
  -5.10429144e-02 -1.19956851e-01 -6.33367836e-01  2.59642005e-01
  -2.84606606e-01  4.34400797e-01  2.50990987e-01  7.65067995e-01
   4.49591994e-01  7.30818987e-01  3.21459436e+00  9.69375312e-01
   1.23549688e+00  2.70643055e-01  2.34755898e+00  9.32763755e-01
   1.46249104e+00  1.07023954e+00  2.46042013e+00  1.44383121e+00
   1.40553844e+00  1.71543527e+00  1.49194574e+00  1.64611411e+00
   8.84159684e-01  1.86987662e+00  1.07190204e+00  4.18888628e-01
   1.10931444e+00  1.81744063e+00  1.71556044e+00  1.40836012e+00
   5.23988307e-01  1.39254308e+00  3.34685802e-01  1.26534963e+00
   7.78617024e-01  1.94506526e+00  6.45797551e-01  1.77644753e+00
   3.13969731e-01  2.61747289e+00  5.40422082e-01  1.08137691e+00
   3.23589444e-01  1.47016692e+00  1.95829630e+00  8.69517148e-01
   1.91240597e+00  4.00068462e-01  1.77845180e+00  1.83818293e+00
   4.47454274e-01  1.33621264e+00  4.63787198e-01  5.55832028e-01
   1.39316845e+00  1.39668763e+00  1.35529196e+00  1.05596375e+00
   1.64922094e+00  9.37243819e-01  6.73668861e-01  1.79238939e+00
   1.65811288e+00  1.39525414e+00  1.54350162e+00  1.01474643e+00
   9.06774402e-02  7.49125004e+00  6.58408022e+00  6.37423515e+00
   5.85721540e+00  6.34674358e+00  7.10089922e+00  7.28491068e+00
   6.15819216e+00  6.80870962e+00  7.16101551e+00  6.59790611e+00
   6.45100832e+00  6.04258442e+00  6.54221487e+00  6.06598186e+00
   6.39206886e+00  7.48766375e+00  6.77958012e+00  6.48520851e+00
   7.22144079e+00  7.00765562e+00  5.84939861e+00  7.72898674e+00
   7.99545383e+00  6.51249933e+00  6.78063869e+00  6.71807671e+00
   6.24827766e+00  7.51388741e+00  7.03434038e+00  6.31308556e+00
   6.27673817e+00  6.60152960e+00  6.57876253e+00  5.49686909e+00
   6.81728554e+00  7.25032091e+00  6.25990200e+00  6.34269524e+00
   7.43909836e+00  7.07913017e+00  7.26623726e+00  7.72579908e+00
   6.58009815e+00  5.66794157e+00  7.36708498e+00  6.32780361e+00
   5.65686703e+00  7.15727425e+00  6.53081322e+00  6.84583378e+00
   6.68986511e+00  6.46415758e+00  7.51906872e+00  6.55038595e+00
   6.80488539e+00  6.39213657e+00  7.00271130e+00  6.96301222e+00
   3.00376773e-01  8.97784650e-01  7.10696340e-01  1.32066631e+00
   3.56116533e-01  1.56882751e+00  1.02382481e+00  8.68491232e-01
   5.47161341e-01  4.76514935e-01  1.89655149e+00  1.17346811e+00
   2.87703156e-01  1.14389050e+00  5.21574676e-01  8.60475779e-01
   1.51250255e+00  1.03153694e+00  1.45506942e+00  8.48824859e-01
   9.32979643e-01  5.14109552e-01  1.80734503e+00  3.03569555e-01
   8.20362091e-01  2.00051737e+00  1.58383846e+00  1.38399160e+00
   1.58741522e+00  8.18081737e-01  1.35967302e+00  2.24506092e+00
   7.32889771e-01  6.01672828e-01  9.20425177e-01  5.41204154e-01
   5.23969829e-01  9.31846261e-01  8.71951699e-01  4.57485080e-01
   1.57114577e+00  1.08527052e+00  2.03789830e+00  1.42984223e+00
   9.76663828e-01  9.51905966e-01  5.83437979e-01  8.63991022e-01
   1.42270255e+00  6.28751397e-01  3.61252129e-01  3.15987945e-01
   1.15068591e+00  3.61785591e-01  1.43923450e+00  5.53461075e-01
   7.92205036e-01  3.40565205e-01  1.96540678e+00  7.86422133e-01
  -4.36454010e+00  3.06266308e+00 -8.66310978e+00]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 10:03:45.721422
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.3247
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 10:03:45.725725
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    9105.4
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 10:03:45.729289
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   95.6426
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 10:03:45.732721
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -814.453
metric_name                                             r2_score
Name: 7, dtype: object,) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon/fb_prophet.py'},) 
  ('#### Setup Model   ##############################################',) 
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fe752cb7710> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  ({'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, FileNotFoundError(2, "File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist")) 
  ("### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, KeyError('model_uri',)) 
  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/',) 
  (                     date_run  ...            metric_name
0  2020-05-09 10:03:37.569820  ...    mean_absolute_error
1  2020-05-09 10:03:37.574797  ...     mean_squared_error
2  2020-05-09 10:03:37.580970  ...  median_absolute_error
3  2020-05-09 10:03:37.587082  ...               r2_score
4  2020-05-09 10:03:45.721422  ...    mean_absolute_error
5  2020-05-09 10:03:45.725725  ...     mean_squared_error
6  2020-05-09 10:03:45.729289  ...  median_absolute_error
7  2020-05-09 10:03:45.732721  ...               r2_score

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

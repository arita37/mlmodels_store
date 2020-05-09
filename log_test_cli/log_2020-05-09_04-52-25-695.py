  ('/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json',) 
  ('test_cli', 'GITHUB_REPOSITORT', 'GITHUB_SHA') 
  ('Running command', 'test_cli') 
  ('# Testing Command Line System  ',) 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/e9ac15b3b78dd74a306579a71fe3b075fce4b97e', 'url_branch_file': 'https://github.com/{repo}/blob/{branch}/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'e9ac15b3b78dd74a306579a71fe3b075fce4b97e', 'workflow': 'test_cli'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_cli

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/e9ac15b3b78dd74a306579a71fe3b075fce4b97e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/e9ac15b3b78dd74a306579a71fe3b075fce4b97e

 ************************************************************************************************************************
  ('/home/runner/work/mlmodels/mlmodels/mlmodels/../README_usage_CLI.md',) 
  (['# Comand Line tools :\n', '```bash\n', '- ml_models    :  Running model training\n', '- ml_optim     :  Hyper-parameter search\n', '- ml_test      :  Testing for developpers.\n'],) 





 ************************************************************************************************************************
usage: ml_models [-h] [--config_file CONFIG_FILE] [--config_mode CONFIG_MODE]
                 [--log_file LOG_FILE] [--do DO] [--folder FOLDER] [-p PATH]
                 [--model_uri MODEL_URI] [--load_folder LOAD_FOLDER]
                 [--dataname DATANAME] [--save_folder SAVE_FOLDER]
ml_models: error: argument --do: expected one argument
ml_models --do  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
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
ml_models --do init  --path ztest/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
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
ml_models --do model_list  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
generate_config
  ('ztest/',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
ztest/model_tf-1_lstm_config.json
ml_models  --do generate_config  --model_uri model_tf.1_lstm  --save_folder "ztest/"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
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
ml_models --do fit     --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
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
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
test
  ('#### Module init   ############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
  (<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>,) 
  ('#### Loading params   ##############################################',) 
  ('############# Data, Params preparation   #################',) 
  ('#### Model init   ############################################',) 
  (<mlmodels.model_tf.1_lstm.Model object at 0x7f53fefa67f0>,) 
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
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [-1.20168552e-01  2.17859931e-02  9.22081247e-03  1.58468544e-01
   1.87125489e-01 -7.41769560e-04]
 [-6.37133271e-02  2.06016377e-01  4.25757803e-02  4.73136008e-02
  -6.43151179e-02 -2.85155438e-02]
 [ 1.62436977e-01 -2.54203267e-02  1.30855396e-01  2.64749855e-01
  -1.77270710e-01  3.86429131e-01]
 [-1.65232658e-01 -7.84776174e-04  1.50763452e-01  1.10971309e-01
   4.50931340e-02  7.73295686e-02]
 [-2.31935516e-01  1.58639669e-01  7.44230390e-01  3.27609032e-01
  -1.83046103e-01  1.62486255e-01]
 [ 2.78882325e-01 -1.24082556e-02  3.99598569e-01  4.10360768e-02
   1.78861827e-01 -2.22478703e-01]
 [ 1.10869676e-01  1.42878935e-01  4.86888289e-01  3.58436733e-01
  -5.03724217e-02 -4.74039286e-01]
 [ 5.06986454e-02  3.82153988e-01  4.84762400e-01 -1.20986737e-01
   1.54436126e-01  5.38053393e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]]
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
{'loss': 0.5491403937339783, 'loss_history': []}
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
{'loss': 0.4855451062321663, 'loss_history': []}
  ('#### Plot   ########################################################',) 
  ('#### Save/Load   ###################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
ml_models  --do test  --model_uri model_tf.1_lstm  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
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
ml_models --do test  --model_uri "ztest/mycustom/my_lstm.py"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
usage: ml_optim [-h] [--config_file CONFIG_FILE] [--config_mode CONFIG_MODE]
                [--log_file LOG_FILE] [--do DO] [--model_uri MODEL_URI]
                [--ntrials NTRIALS]
ml_optim: error: argument --do: expected one argument
Deprecaton set to False
ml_optim --do  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
Deprecaton set to False
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 387, in main
    optim_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 245, in optim_cli
    js = json.load(open(config_file, mode='r'))  # Config
FileNotFoundError: [Errno 2] No such file or directory: 'optim_config.json'
ml_optim --do search  --config_file optim_config.json  --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
Deprecaton set to False
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 387, in main
    optim_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 245, in optim_cli
    js = json.load(open(config_file, mode='r'))  # Config
FileNotFoundError: [Errno 2] No such file or directory: 'optim_config_prune.json'
ml_optim --do search  --config_file optim_config_prune.json   --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
Deprecaton set to False
  ({'model_uri': 'model_tf.1_lstm', 'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}, {'engine': 'optuna', 'method': 'prune', 'ntrials': 5}, {'engine_pars': {'engine': 'optuna', 'method': 'normal', 'ntrials': 2, 'metric_target': 'loss'}, 'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}, 'num_layers': {'type': 'int', 'init': 2, 'range': [2, 4]}, 'size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'output_size': {'type': 'int', 'init': 6, 'range': [6, 6]}, 'size_layer': {'type': 'categorical', 'value': [128, 256]}, 'timestep': {'type': 'categorical', 'value': [5]}, 'epoch': {'type': 'categorical', 'value': [2]}}) 
  (<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>,) 
  ('###### Hyper-optimization through study   ##################################',) 
  ('check', <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year_small.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}) 
[32m[I 2020-05-09 04:52:58,954][0m Finished trial#0 resulted in value: 0.30504128336906433. Current best value is 0.30504128336906433 with parameters: {'learning_rate': 0.008596081295043875, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
[32m[I 2020-05-09 04:53:01,302][0m Finished trial#1 resulted in value: 2.067322224378586. Current best value is 0.30504128336906433 with parameters: {'learning_rate': 0.008596081295043875, 'num_layers': 2, 'size': 6, 'output_size': 6, 'size_layer': 128, 'timestep': 5, 'epoch': 2}.[0m
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
ml_optim --do test   --model_uri model_tf.1_lstm.py   --ntrials 2  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
  ('dataset/json/benchmark.json',) 
  ('Custom benchmark',) 
  (['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'],) 
  ('Model List', [{'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}}]) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon/fb_prophet.py'},) 
  ('#### Setup Model   ##############################################',) 
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa104ce53c8> <class 'mlmodels.model_gluon.fb_prophet.Model'>
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa0f9fe1128> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 349462.9062
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 209018.0625
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 104169.2109
Epoch 4/10

1/1 [==============================] - 0s 111ms/step - loss: 48468.1016
Epoch 5/10

1/1 [==============================] - 0s 128ms/step - loss: 24261.7812
Epoch 6/10

1/1 [==============================] - 0s 99ms/step - loss: 13905.8467
Epoch 7/10

1/1 [==============================] - 0s 100ms/step - loss: 8963.7246
Epoch 8/10

1/1 [==============================] - 0s 98ms/step - loss: 6305.2837
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 4755.2681
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 3800.1648
  ('#### Inference Need return ypred, ytrue #########################',) 
[[  0.57484853  12.540343    13.791054    11.379357    11.135535
   12.753922    11.778143    10.920987    11.567003    10.645431
   10.255331    10.335455    12.2543      10.669248    12.487369
   11.857927    11.5196905    9.41783     11.168417    12.109068
   11.280883    13.432225    11.735692    10.719542     9.407493
    9.998613    11.428013    10.400658    11.032985    12.2668915
   10.307319     9.585716    12.053377    13.806079    13.28979
   10.401342    11.157394     9.332702    11.308778    13.214678
   12.176428    13.29265     10.830502    11.332081    11.196954
   12.131217    12.570146    11.074884    13.971124    11.647381
   10.168422    12.332244    11.270934    11.057042    11.827033
   10.6833515   12.002796    10.69009     13.02875     13.649117
   -0.35357213  -2.8782806    0.45897448   0.2885792    2.4087105
   -1.2022508   -0.8637953   -1.4196022   -1.0394154   -1.1068106
   -2.8795867   -0.54449666   0.29338104  -0.81722504   1.7973486
   -1.1496598    0.39218977   1.3500874   -1.0844041    0.01671696
   -0.9098736    0.7864959   -0.60774213   0.35428652   0.6476858
   -1.892678    -2.2883248   -0.47353113  -1.5569246   -1.2291805
    0.12138264   1.9599032   -3.204442    -2.358306     0.26067296
    0.255773    -2.45647     -0.45511925   1.0292703   -0.77315307
    0.11895847   1.4113319   -2.4002566    0.51620036  -0.65603805
    0.8899271    0.47803026   1.1890596    0.53327566   1.1254294
   -0.35501647   0.32629603  -1.615197     0.10464197  -0.7326643
    0.19402957   0.66988236   0.4209656   -1.5817401    1.3408138
    1.5416185    0.48245895  -0.20045173  -1.6979369   -0.40105247
   -0.01829499  -1.1328433   -0.09895852  -0.07247937   1.3019397
    0.04445933   1.1471674   -1.5880958    1.141809    -0.45207372
    0.9466709   -1.4264119    0.54909414   1.4113266    1.3565979
   -0.43015265  -0.4622686    1.4078459   -0.5534329   -1.7153969
    1.757444     1.4557202   -1.9169431   -1.4084401    0.02652508
    1.3530123    1.3195095    1.997843     0.9641152   -0.7330423
    0.6117438   -0.2669106   -1.1222054    0.68642306   1.0419292
   -1.2818915   -1.0824801   -1.5549076    1.2214057   -1.4548357
    0.43230182  -0.73369765   0.7803857   -0.4178477   -1.1905724
    0.13531333  -0.9397676   -0.62332916  -0.95208585   1.1286148
   -1.0830534   -0.20550966  -1.7485626   -0.20239586   0.11985604
    0.79402226  11.455876    11.544367    12.2172165    9.579135
   11.163307    13.403428    11.408624    10.94694     11.843853
   11.971721    10.676526    12.173137    13.058175    10.326743
   11.500245    11.48469     11.68151     11.310973    11.956506
   11.208244    10.938791    11.657384     9.466599    11.19863
   12.733138    11.142545     9.832767    11.012538    11.879318
   11.084684     9.916704    10.470935    11.593917    12.50991
   10.8804245   11.124172    12.942919    10.254072    10.41427
   12.444664    10.177469    10.731547    12.263142    11.754775
   11.053695    10.996803    12.073397    12.687369     9.562217
    9.635159    12.442032    10.065482     9.383167    10.359768
    9.13425     12.083328    10.918387    13.359117    11.782618
    0.7948898    0.5632381    2.3502822    0.60774845   0.8728591
    0.6417116    0.4171946    0.27756917   1.2583965    1.7596593
    0.8236061    0.6937362    3.5584188    0.90500146   0.9564273
    1.9340208    0.04025352   2.821724     1.1968905    2.0127182
    0.08341092   0.42338115   2.6227727    1.9536278    1.9003779
    3.5067573    0.38179302   1.5144548    0.7509184    1.0041707
    1.6887562    3.1207542    0.50380063   0.28041804   0.18723261
    0.25010633   0.71392435   2.3699667    1.5794406    1.1476814
    2.2349052    3.97408      0.40559798   1.1213264    2.4801064
    1.3698153    0.51493835   0.25772125   0.41406375   1.5299358
    0.4128654    0.3363046    2.4696693    2.146883     1.6331503
    1.3634853    1.3248193    0.7271608    3.344068     0.41280538
    1.1815896    1.1311139    2.9371543    3.1373587    0.88487494
    0.76081556   1.201499     0.3543856    0.8582303    0.16782612
    2.0720751    2.011055     0.30721408   0.73504394   2.5963135
    1.1871946    0.48066628   0.20586044   0.7657651    0.9385804
    0.45134312   0.9178303    0.37534922   0.17259288   2.0165782
    1.369699     2.924056     0.44569206   3.017437     1.8802841
    2.7746196    2.1123676    1.530743     0.59735346   0.951834
    0.50388944   0.358698     0.4376725    0.75329745   2.67232
    0.721063     1.1592636    2.0385394    0.39978367   0.40153402
    0.70937127   0.99628776   0.3954038    0.11054587   0.8321039
    3.1535988    1.2427074    0.9846039    0.14234406   1.7132815
    0.3353842    0.44560695   0.35522985   0.98768806   1.4785061
   10.605106   -10.959556   -11.313241  ]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 04:53:15.888344
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   91.0853
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 04:53:15.892442
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8319.22
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 04:53:15.895802
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                    91.909
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 04:53:15.899190
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -744.045
metric_name                                             r2_score
Name: 3, dtype: object,) 
  ("### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256},) 
  ('#### Setup Model   ##############################################',) 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140329031652352
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140326521294016
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140326521294520
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140326520905968
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140326520906472
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140326520906976
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa104ce5438> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.564795
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.529647
grad_step = 000002, loss = 0.502145
grad_step = 000003, loss = 0.471028
grad_step = 000004, loss = 0.438683
grad_step = 000005, loss = 0.412854
grad_step = 000006, loss = 0.404752
grad_step = 000007, loss = 0.398255
grad_step = 000008, loss = 0.376239
grad_step = 000009, loss = 0.355150
grad_step = 000010, loss = 0.342192
grad_step = 000011, loss = 0.333412
grad_step = 000012, loss = 0.324068
grad_step = 000013, loss = 0.312711
grad_step = 000014, loss = 0.300468
grad_step = 000015, loss = 0.288540
grad_step = 000016, loss = 0.276787
grad_step = 000017, loss = 0.266172
grad_step = 000018, loss = 0.256411
grad_step = 000019, loss = 0.245973
grad_step = 000020, loss = 0.234994
grad_step = 000021, loss = 0.224927
grad_step = 000022, loss = 0.216022
grad_step = 000023, loss = 0.207228
grad_step = 000024, loss = 0.197957
grad_step = 000025, loss = 0.188766
grad_step = 000026, loss = 0.180198
grad_step = 000027, loss = 0.172148
grad_step = 000028, loss = 0.164266
grad_step = 000029, loss = 0.156356
grad_step = 000030, loss = 0.148420
grad_step = 000031, loss = 0.140831
grad_step = 000032, loss = 0.133966
grad_step = 000033, loss = 0.127476
grad_step = 000034, loss = 0.120914
grad_step = 000035, loss = 0.114468
grad_step = 000036, loss = 0.108423
grad_step = 000037, loss = 0.102617
grad_step = 000038, loss = 0.096930
grad_step = 000039, loss = 0.091412
grad_step = 000040, loss = 0.086163
grad_step = 000041, loss = 0.081253
grad_step = 000042, loss = 0.076546
grad_step = 000043, loss = 0.071875
grad_step = 000044, loss = 0.067417
grad_step = 000045, loss = 0.063272
grad_step = 000046, loss = 0.059278
grad_step = 000047, loss = 0.055429
grad_step = 000048, loss = 0.051797
grad_step = 000049, loss = 0.048359
grad_step = 000050, loss = 0.045099
grad_step = 000051, loss = 0.041970
grad_step = 000052, loss = 0.039025
grad_step = 000053, loss = 0.036295
grad_step = 000054, loss = 0.033670
grad_step = 000055, loss = 0.031180
grad_step = 000056, loss = 0.028859
grad_step = 000057, loss = 0.026672
grad_step = 000058, loss = 0.024626
grad_step = 000059, loss = 0.022702
grad_step = 000060, loss = 0.020922
grad_step = 000061, loss = 0.019265
grad_step = 000062, loss = 0.017702
grad_step = 000063, loss = 0.016268
grad_step = 000064, loss = 0.014941
grad_step = 000065, loss = 0.013692
grad_step = 000066, loss = 0.012537
grad_step = 000067, loss = 0.011493
grad_step = 000068, loss = 0.010522
grad_step = 000069, loss = 0.009642
grad_step = 000070, loss = 0.008857
grad_step = 000071, loss = 0.008134
grad_step = 000072, loss = 0.007490
grad_step = 000073, loss = 0.006900
grad_step = 000074, loss = 0.006375
grad_step = 000075, loss = 0.005897
grad_step = 000076, loss = 0.005478
grad_step = 000077, loss = 0.005097
grad_step = 000078, loss = 0.004757
grad_step = 000079, loss = 0.004452
grad_step = 000080, loss = 0.004178
grad_step = 000081, loss = 0.003933
grad_step = 000082, loss = 0.003720
grad_step = 000083, loss = 0.003528
grad_step = 000084, loss = 0.003359
grad_step = 000085, loss = 0.003205
grad_step = 000086, loss = 0.003069
grad_step = 000087, loss = 0.002946
grad_step = 000088, loss = 0.002842
grad_step = 000089, loss = 0.002746
grad_step = 000090, loss = 0.002665
grad_step = 000091, loss = 0.002592
grad_step = 000092, loss = 0.002527
grad_step = 000093, loss = 0.002472
grad_step = 000094, loss = 0.002422
grad_step = 000095, loss = 0.002381
grad_step = 000096, loss = 0.002345
grad_step = 000097, loss = 0.002314
grad_step = 000098, loss = 0.002287
grad_step = 000099, loss = 0.002264
grad_step = 000100, loss = 0.002244
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002228
grad_step = 000102, loss = 0.002213
grad_step = 000103, loss = 0.002201
grad_step = 000104, loss = 0.002190
grad_step = 000105, loss = 0.002180
grad_step = 000106, loss = 0.002172
grad_step = 000107, loss = 0.002164
grad_step = 000108, loss = 0.002157
grad_step = 000109, loss = 0.002151
grad_step = 000110, loss = 0.002145
grad_step = 000111, loss = 0.002139
grad_step = 000112, loss = 0.002133
grad_step = 000113, loss = 0.002127
grad_step = 000114, loss = 0.002121
grad_step = 000115, loss = 0.002115
grad_step = 000116, loss = 0.002109
grad_step = 000117, loss = 0.002103
grad_step = 000118, loss = 0.002097
grad_step = 000119, loss = 0.002090
grad_step = 000120, loss = 0.002084
grad_step = 000121, loss = 0.002077
grad_step = 000122, loss = 0.002070
grad_step = 000123, loss = 0.002063
grad_step = 000124, loss = 0.002055
grad_step = 000125, loss = 0.002047
grad_step = 000126, loss = 0.002039
grad_step = 000127, loss = 0.002031
grad_step = 000128, loss = 0.002023
grad_step = 000129, loss = 0.002015
grad_step = 000130, loss = 0.002008
grad_step = 000131, loss = 0.001996
grad_step = 000132, loss = 0.001990
grad_step = 000133, loss = 0.001981
grad_step = 000134, loss = 0.001972
grad_step = 000135, loss = 0.001962
grad_step = 000136, loss = 0.001952
grad_step = 000137, loss = 0.001943
grad_step = 000138, loss = 0.001935
grad_step = 000139, loss = 0.001928
grad_step = 000140, loss = 0.001916
grad_step = 000141, loss = 0.001908
grad_step = 000142, loss = 0.001901
grad_step = 000143, loss = 0.001897
grad_step = 000144, loss = 0.001896
grad_step = 000145, loss = 0.001882
grad_step = 000146, loss = 0.001881
grad_step = 000147, loss = 0.001881
grad_step = 000148, loss = 0.001885
grad_step = 000149, loss = 0.001850
grad_step = 000150, loss = 0.001823
grad_step = 000151, loss = 0.001804
grad_step = 000152, loss = 0.001795
grad_step = 000153, loss = 0.001784
grad_step = 000154, loss = 0.001782
grad_step = 000155, loss = 0.001816
grad_step = 000156, loss = 0.001917
grad_step = 000157, loss = 0.002172
grad_step = 000158, loss = 0.001975
grad_step = 000159, loss = 0.001768
grad_step = 000160, loss = 0.001892
grad_step = 000161, loss = 0.001905
grad_step = 000162, loss = 0.001838
grad_step = 000163, loss = 0.001778
grad_step = 000164, loss = 0.001813
grad_step = 000165, loss = 0.001817
grad_step = 000166, loss = 0.001729
grad_step = 000167, loss = 0.001801
grad_step = 000168, loss = 0.001807
grad_step = 000169, loss = 0.001705
grad_step = 000170, loss = 0.001726
grad_step = 000171, loss = 0.001832
grad_step = 000172, loss = 0.001748
grad_step = 000173, loss = 0.001639
grad_step = 000174, loss = 0.001738
grad_step = 000175, loss = 0.001733
grad_step = 000176, loss = 0.001691
grad_step = 000177, loss = 0.001650
grad_step = 000178, loss = 0.001650
grad_step = 000179, loss = 0.001734
grad_step = 000180, loss = 0.001705
grad_step = 000181, loss = 0.001651
grad_step = 000182, loss = 0.001613
grad_step = 000183, loss = 0.001620
grad_step = 000184, loss = 0.001687
grad_step = 000185, loss = 0.001676
grad_step = 000186, loss = 0.001644
grad_step = 000187, loss = 0.001585
grad_step = 000188, loss = 0.001562
grad_step = 000189, loss = 0.001614
grad_step = 000190, loss = 0.001622
grad_step = 000191, loss = 0.001613
grad_step = 000192, loss = 0.001581
grad_step = 000193, loss = 0.001537
grad_step = 000194, loss = 0.001536
grad_step = 000195, loss = 0.001530
grad_step = 000196, loss = 0.001550
grad_step = 000197, loss = 0.001574
grad_step = 000198, loss = 0.001612
grad_step = 000199, loss = 0.001635
grad_step = 000200, loss = 0.001634
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001546
grad_step = 000202, loss = 0.001508
grad_step = 000203, loss = 0.001544
grad_step = 000204, loss = 0.001564
grad_step = 000205, loss = 0.001507
grad_step = 000206, loss = 0.001492
grad_step = 000207, loss = 0.001523
grad_step = 000208, loss = 0.001565
grad_step = 000209, loss = 0.001549
grad_step = 000210, loss = 0.001505
grad_step = 000211, loss = 0.001469
grad_step = 000212, loss = 0.001476
grad_step = 000213, loss = 0.001513
grad_step = 000214, loss = 0.001523
grad_step = 000215, loss = 0.001492
grad_step = 000216, loss = 0.001462
grad_step = 000217, loss = 0.001439
grad_step = 000218, loss = 0.001445
grad_step = 000219, loss = 0.001460
grad_step = 000220, loss = 0.001493
grad_step = 000221, loss = 0.001526
grad_step = 000222, loss = 0.001589
grad_step = 000223, loss = 0.001556
grad_step = 000224, loss = 0.001475
grad_step = 000225, loss = 0.001414
grad_step = 000226, loss = 0.001437
grad_step = 000227, loss = 0.001469
grad_step = 000228, loss = 0.001422
grad_step = 000229, loss = 0.001395
grad_step = 000230, loss = 0.001417
grad_step = 000231, loss = 0.001423
grad_step = 000232, loss = 0.001402
grad_step = 000233, loss = 0.001380
grad_step = 000234, loss = 0.001373
grad_step = 000235, loss = 0.001376
grad_step = 000236, loss = 0.001391
grad_step = 000237, loss = 0.001435
grad_step = 000238, loss = 0.001472
grad_step = 000239, loss = 0.001556
grad_step = 000240, loss = 0.001483
grad_step = 000241, loss = 0.001369
grad_step = 000242, loss = 0.001349
grad_step = 000243, loss = 0.001413
grad_step = 000244, loss = 0.001425
grad_step = 000245, loss = 0.001341
grad_step = 000246, loss = 0.001336
grad_step = 000247, loss = 0.001388
grad_step = 000248, loss = 0.001358
grad_step = 000249, loss = 0.001303
grad_step = 000250, loss = 0.001308
grad_step = 000251, loss = 0.001337
grad_step = 000252, loss = 0.001373
grad_step = 000253, loss = 0.001360
grad_step = 000254, loss = 0.001375
grad_step = 000255, loss = 0.001316
grad_step = 000256, loss = 0.001286
grad_step = 000257, loss = 0.001263
grad_step = 000258, loss = 0.001265
grad_step = 000259, loss = 0.001296
grad_step = 000260, loss = 0.001291
grad_step = 000261, loss = 0.001276
grad_step = 000262, loss = 0.001246
grad_step = 000263, loss = 0.001227
grad_step = 000264, loss = 0.001231
grad_step = 000265, loss = 0.001252
grad_step = 000266, loss = 0.001299
grad_step = 000267, loss = 0.001329
grad_step = 000268, loss = 0.001427
grad_step = 000269, loss = 0.001350
grad_step = 000270, loss = 0.001237
grad_step = 000271, loss = 0.001204
grad_step = 000272, loss = 0.001277
grad_step = 000273, loss = 0.001267
grad_step = 000274, loss = 0.001184
grad_step = 000275, loss = 0.001196
grad_step = 000276, loss = 0.001246
grad_step = 000277, loss = 0.001208
grad_step = 000278, loss = 0.001167
grad_step = 000279, loss = 0.001152
grad_step = 000280, loss = 0.001175
grad_step = 000281, loss = 0.001208
grad_step = 000282, loss = 0.001195
grad_step = 000283, loss = 0.001176
grad_step = 000284, loss = 0.001131
grad_step = 000285, loss = 0.001114
grad_step = 000286, loss = 0.001112
grad_step = 000287, loss = 0.001116
grad_step = 000288, loss = 0.001126
grad_step = 000289, loss = 0.001117
grad_step = 000290, loss = 0.001108
grad_step = 000291, loss = 0.001081
grad_step = 000292, loss = 0.001062
grad_step = 000293, loss = 0.001055
grad_step = 000294, loss = 0.001058
grad_step = 000295, loss = 0.001061
grad_step = 000296, loss = 0.001067
grad_step = 000297, loss = 0.001092
grad_step = 000298, loss = 0.001132
grad_step = 000299, loss = 0.001234
grad_step = 000300, loss = 0.001326
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001452
grad_step = 000302, loss = 0.001286
grad_step = 000303, loss = 0.001072
grad_step = 000304, loss = 0.001012
grad_step = 000305, loss = 0.001162
grad_step = 000306, loss = 0.001233
grad_step = 000307, loss = 0.001024
grad_step = 000308, loss = 0.001024
grad_step = 000309, loss = 0.001194
grad_step = 000310, loss = 0.001074
grad_step = 000311, loss = 0.000967
grad_step = 000312, loss = 0.001012
grad_step = 000313, loss = 0.001058
grad_step = 000314, loss = 0.001028
grad_step = 000315, loss = 0.000948
grad_step = 000316, loss = 0.000991
grad_step = 000317, loss = 0.001058
grad_step = 000318, loss = 0.000948
grad_step = 000319, loss = 0.000929
grad_step = 000320, loss = 0.000977
grad_step = 000321, loss = 0.000963
grad_step = 000322, loss = 0.000923
grad_step = 000323, loss = 0.000893
grad_step = 000324, loss = 0.000908
grad_step = 000325, loss = 0.000935
grad_step = 000326, loss = 0.000901
grad_step = 000327, loss = 0.000872
grad_step = 000328, loss = 0.000862
grad_step = 000329, loss = 0.000880
grad_step = 000330, loss = 0.000901
grad_step = 000331, loss = 0.000885
grad_step = 000332, loss = 0.000852
grad_step = 000333, loss = 0.000834
grad_step = 000334, loss = 0.000825
grad_step = 000335, loss = 0.000829
grad_step = 000336, loss = 0.000826
grad_step = 000337, loss = 0.000830
grad_step = 000338, loss = 0.000828
grad_step = 000339, loss = 0.000831
grad_step = 000340, loss = 0.000821
grad_step = 000341, loss = 0.000820
grad_step = 000342, loss = 0.000808
grad_step = 000343, loss = 0.000811
grad_step = 000344, loss = 0.000809
grad_step = 000345, loss = 0.000819
grad_step = 000346, loss = 0.000820
grad_step = 000347, loss = 0.000846
grad_step = 000348, loss = 0.000834
grad_step = 000349, loss = 0.000844
grad_step = 000350, loss = 0.000820
grad_step = 000351, loss = 0.000811
grad_step = 000352, loss = 0.000774
grad_step = 000353, loss = 0.000747
grad_step = 000354, loss = 0.000730
grad_step = 000355, loss = 0.000728
grad_step = 000356, loss = 0.000736
grad_step = 000357, loss = 0.000745
grad_step = 000358, loss = 0.000763
grad_step = 000359, loss = 0.000771
grad_step = 000360, loss = 0.000796
grad_step = 000361, loss = 0.000786
grad_step = 000362, loss = 0.000799
grad_step = 000363, loss = 0.000766
grad_step = 000364, loss = 0.000747
grad_step = 000365, loss = 0.000705
grad_step = 000366, loss = 0.000681
grad_step = 000367, loss = 0.000672
grad_step = 000368, loss = 0.000679
grad_step = 000369, loss = 0.000695
grad_step = 000370, loss = 0.000710
grad_step = 000371, loss = 0.000738
grad_step = 000372, loss = 0.000736
grad_step = 000373, loss = 0.000752
grad_step = 000374, loss = 0.000726
grad_step = 000375, loss = 0.000723
grad_step = 000376, loss = 0.000688
grad_step = 000377, loss = 0.000662
grad_step = 000378, loss = 0.000639
grad_step = 000379, loss = 0.000628
grad_step = 000380, loss = 0.000628
grad_step = 000381, loss = 0.000636
grad_step = 000382, loss = 0.000647
grad_step = 000383, loss = 0.000659
grad_step = 000384, loss = 0.000680
grad_step = 000385, loss = 0.000691
grad_step = 000386, loss = 0.000744
grad_step = 000387, loss = 0.000752
grad_step = 000388, loss = 0.000810
grad_step = 000389, loss = 0.000740
grad_step = 000390, loss = 0.000702
grad_step = 000391, loss = 0.000616
grad_step = 000392, loss = 0.000593
grad_step = 000393, loss = 0.000625
grad_step = 000394, loss = 0.000661
grad_step = 000395, loss = 0.000686
grad_step = 000396, loss = 0.000622
grad_step = 000397, loss = 0.000594
grad_step = 000398, loss = 0.000595
grad_step = 000399, loss = 0.000624
grad_step = 000400, loss = 0.000623
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000605
grad_step = 000402, loss = 0.000593
grad_step = 000403, loss = 0.000579
grad_step = 000404, loss = 0.000592
grad_step = 000405, loss = 0.000588
grad_step = 000406, loss = 0.000590
grad_step = 000407, loss = 0.000565
grad_step = 000408, loss = 0.000544
grad_step = 000409, loss = 0.000547
grad_step = 000410, loss = 0.000551
grad_step = 000411, loss = 0.000570
grad_step = 000412, loss = 0.000574
grad_step = 000413, loss = 0.000574
grad_step = 000414, loss = 0.000588
grad_step = 000415, loss = 0.000648
grad_step = 000416, loss = 0.000725
grad_step = 000417, loss = 0.000983
grad_step = 000418, loss = 0.000942
grad_step = 000419, loss = 0.000939
grad_step = 000420, loss = 0.000626
grad_step = 000421, loss = 0.000598
grad_step = 000422, loss = 0.000777
grad_step = 000423, loss = 0.000743
grad_step = 000424, loss = 0.000629
grad_step = 000425, loss = 0.000552
grad_step = 000426, loss = 0.000637
grad_step = 000427, loss = 0.000722
grad_step = 000428, loss = 0.000604
grad_step = 000429, loss = 0.000557
grad_step = 000430, loss = 0.000593
grad_step = 000431, loss = 0.000614
grad_step = 000432, loss = 0.000589
grad_step = 000433, loss = 0.000495
grad_step = 000434, loss = 0.000561
grad_step = 000435, loss = 0.000622
grad_step = 000436, loss = 0.000548
grad_step = 000437, loss = 0.000484
grad_step = 000438, loss = 0.000518
grad_step = 000439, loss = 0.000555
grad_step = 000440, loss = 0.000521
grad_step = 000441, loss = 0.000467
grad_step = 000442, loss = 0.000501
grad_step = 000443, loss = 0.000533
grad_step = 000444, loss = 0.000493
grad_step = 000445, loss = 0.000458
grad_step = 000446, loss = 0.000482
grad_step = 000447, loss = 0.000496
grad_step = 000448, loss = 0.000480
grad_step = 000449, loss = 0.000453
grad_step = 000450, loss = 0.000470
grad_step = 000451, loss = 0.000481
grad_step = 000452, loss = 0.000455
grad_step = 000453, loss = 0.000442
grad_step = 000454, loss = 0.000455
grad_step = 000455, loss = 0.000461
grad_step = 000456, loss = 0.000448
grad_step = 000457, loss = 0.000433
grad_step = 000458, loss = 0.000441
grad_step = 000459, loss = 0.000448
grad_step = 000460, loss = 0.000437
grad_step = 000461, loss = 0.000428
grad_step = 000462, loss = 0.000431
grad_step = 000463, loss = 0.000436
grad_step = 000464, loss = 0.000431
grad_step = 000465, loss = 0.000423
grad_step = 000466, loss = 0.000426
grad_step = 000467, loss = 0.000438
grad_step = 000468, loss = 0.000441
grad_step = 000469, loss = 0.000454
grad_step = 000470, loss = 0.000472
grad_step = 000471, loss = 0.000525
grad_step = 000472, loss = 0.000557
grad_step = 000473, loss = 0.000634
grad_step = 000474, loss = 0.000583
grad_step = 000475, loss = 0.000558
grad_step = 000476, loss = 0.000453
grad_step = 000477, loss = 0.000406
grad_step = 000478, loss = 0.000433
grad_step = 000479, loss = 0.000475
grad_step = 000480, loss = 0.000502
grad_step = 000481, loss = 0.000459
grad_step = 000482, loss = 0.000420
grad_step = 000483, loss = 0.000399
grad_step = 000484, loss = 0.000405
grad_step = 000485, loss = 0.000432
grad_step = 000486, loss = 0.000443
grad_step = 000487, loss = 0.000440
grad_step = 000488, loss = 0.000417
grad_step = 000489, loss = 0.000396
grad_step = 000490, loss = 0.000388
grad_step = 000491, loss = 0.000392
grad_step = 000492, loss = 0.000401
grad_step = 000493, loss = 0.000408
grad_step = 000494, loss = 0.000411
grad_step = 000495, loss = 0.000402
grad_step = 000496, loss = 0.000392
grad_step = 000497, loss = 0.000382
grad_step = 000498, loss = 0.000378
grad_step = 000499, loss = 0.000379
grad_step = 000500, loss = 0.000383
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000387
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
  (date_run                              2020-05-09 04:53:39.469704
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.234076
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 04:53:39.474782
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.147638
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 04:53:39.481291
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.121033
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 04:53:39.486068
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -1.24342
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
0  2020-05-09 04:53:15.888344  ...    mean_absolute_error
1  2020-05-09 04:53:15.892442  ...     mean_squared_error
2  2020-05-09 04:53:15.895802  ...  median_absolute_error
3  2020-05-09 04:53:15.899190  ...               r2_score
4  2020-05-09 04:53:39.469704  ...    mean_absolute_error
5  2020-05-09 04:53:39.474782  ...     mean_squared_error
6  2020-05-09 04:53:39.481291  ...  median_absolute_error
7  2020-05-09 04:53:39.486068  ...               r2_score

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
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test02/model_list.json  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt





 ************************************************************************************************************************
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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139887847667248
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139886568865632
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139886568415640
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139886568416144
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139886568416648
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139886568417152
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f3a2d77ffd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.586398
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.542098
grad_step = 000002, loss = 0.504112
grad_step = 000003, loss = 0.462463
grad_step = 000004, loss = 0.417505
grad_step = 000005, loss = 0.375193
grad_step = 000006, loss = 0.353470
grad_step = 000007, loss = 0.352683
grad_step = 000008, loss = 0.330642
grad_step = 000009, loss = 0.302421
grad_step = 000010, loss = 0.284550
grad_step = 000011, loss = 0.274582
grad_step = 000012, loss = 0.265826
grad_step = 000013, loss = 0.254796
grad_step = 000014, loss = 0.241299
grad_step = 000015, loss = 0.227024
grad_step = 000016, loss = 0.214034
grad_step = 000017, loss = 0.202619
grad_step = 000018, loss = 0.191412
grad_step = 000019, loss = 0.180061
grad_step = 000020, loss = 0.169312
grad_step = 000021, loss = 0.159454
grad_step = 000022, loss = 0.150372
grad_step = 000023, loss = 0.141638
grad_step = 000024, loss = 0.132764
grad_step = 000025, loss = 0.123686
grad_step = 000026, loss = 0.115002
grad_step = 000027, loss = 0.107481
grad_step = 000028, loss = 0.101054
grad_step = 000029, loss = 0.094681
grad_step = 000030, loss = 0.087894
grad_step = 000031, loss = 0.081245
grad_step = 000032, loss = 0.075333
grad_step = 000033, loss = 0.070073
grad_step = 000034, loss = 0.065065
grad_step = 000035, loss = 0.060125
grad_step = 000036, loss = 0.055425
grad_step = 000037, loss = 0.051217
grad_step = 000038, loss = 0.047455
grad_step = 000039, loss = 0.043870
grad_step = 000040, loss = 0.040383
grad_step = 000041, loss = 0.037136
grad_step = 000042, loss = 0.034206
grad_step = 000043, loss = 0.031478
grad_step = 000044, loss = 0.028873
grad_step = 000045, loss = 0.026463
grad_step = 000046, loss = 0.024343
grad_step = 000047, loss = 0.022436
grad_step = 000048, loss = 0.020580
grad_step = 000049, loss = 0.018777
grad_step = 000050, loss = 0.017163
grad_step = 000051, loss = 0.015780
grad_step = 000052, loss = 0.014537
grad_step = 000053, loss = 0.013356
grad_step = 000054, loss = 0.012261
grad_step = 000055, loss = 0.011286
grad_step = 000056, loss = 0.010394
grad_step = 000057, loss = 0.009548
grad_step = 000058, loss = 0.008790
grad_step = 000059, loss = 0.008154
grad_step = 000060, loss = 0.007584
grad_step = 000061, loss = 0.007020
grad_step = 000062, loss = 0.006488
grad_step = 000063, loss = 0.006038
grad_step = 000064, loss = 0.005660
grad_step = 000065, loss = 0.005306
grad_step = 000066, loss = 0.004970
grad_step = 000067, loss = 0.004676
grad_step = 000068, loss = 0.004419
grad_step = 000069, loss = 0.004179
grad_step = 000070, loss = 0.003959
grad_step = 000071, loss = 0.003769
grad_step = 000072, loss = 0.003597
grad_step = 000073, loss = 0.003435
grad_step = 000074, loss = 0.003292
grad_step = 000075, loss = 0.003170
grad_step = 000076, loss = 0.003055
grad_step = 000077, loss = 0.002946
grad_step = 000078, loss = 0.002852
grad_step = 000079, loss = 0.002773
grad_step = 000080, loss = 0.002698
grad_step = 000081, loss = 0.002628
grad_step = 000082, loss = 0.002567
grad_step = 000083, loss = 0.002514
grad_step = 000084, loss = 0.002465
grad_step = 000085, loss = 0.002422
grad_step = 000086, loss = 0.002387
grad_step = 000087, loss = 0.002356
grad_step = 000088, loss = 0.002326
grad_step = 000089, loss = 0.002300
grad_step = 000090, loss = 0.002280
grad_step = 000091, loss = 0.002263
grad_step = 000092, loss = 0.002246
grad_step = 000093, loss = 0.002233
grad_step = 000094, loss = 0.002222
grad_step = 000095, loss = 0.002212
grad_step = 000096, loss = 0.002203
grad_step = 000097, loss = 0.002196
grad_step = 000098, loss = 0.002190
grad_step = 000099, loss = 0.002184
grad_step = 000100, loss = 0.002180
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002175
grad_step = 000102, loss = 0.002171
grad_step = 000103, loss = 0.002166
grad_step = 000104, loss = 0.002163
grad_step = 000105, loss = 0.002159
grad_step = 000106, loss = 0.002156
grad_step = 000107, loss = 0.002152
grad_step = 000108, loss = 0.002149
grad_step = 000109, loss = 0.002145
grad_step = 000110, loss = 0.002141
grad_step = 000111, loss = 0.002138
grad_step = 000112, loss = 0.002135
grad_step = 000113, loss = 0.002131
grad_step = 000114, loss = 0.002128
grad_step = 000115, loss = 0.002124
grad_step = 000116, loss = 0.002121
grad_step = 000117, loss = 0.002118
grad_step = 000118, loss = 0.002115
grad_step = 000119, loss = 0.002111
grad_step = 000120, loss = 0.002108
grad_step = 000121, loss = 0.002105
grad_step = 000122, loss = 0.002102
grad_step = 000123, loss = 0.002100
grad_step = 000124, loss = 0.002097
grad_step = 000125, loss = 0.002094
grad_step = 000126, loss = 0.002091
grad_step = 000127, loss = 0.002089
grad_step = 000128, loss = 0.002086
grad_step = 000129, loss = 0.002083
grad_step = 000130, loss = 0.002081
grad_step = 000131, loss = 0.002078
grad_step = 000132, loss = 0.002076
grad_step = 000133, loss = 0.002073
grad_step = 000134, loss = 0.002071
grad_step = 000135, loss = 0.002069
grad_step = 000136, loss = 0.002066
grad_step = 000137, loss = 0.002064
grad_step = 000138, loss = 0.002061
grad_step = 000139, loss = 0.002059
grad_step = 000140, loss = 0.002056
grad_step = 000141, loss = 0.002054
grad_step = 000142, loss = 0.002052
grad_step = 000143, loss = 0.002049
grad_step = 000144, loss = 0.002047
grad_step = 000145, loss = 0.002044
grad_step = 000146, loss = 0.002042
grad_step = 000147, loss = 0.002039
grad_step = 000148, loss = 0.002037
grad_step = 000149, loss = 0.002034
grad_step = 000150, loss = 0.002032
grad_step = 000151, loss = 0.002029
grad_step = 000152, loss = 0.002027
grad_step = 000153, loss = 0.002024
grad_step = 000154, loss = 0.002022
grad_step = 000155, loss = 0.002019
grad_step = 000156, loss = 0.002017
grad_step = 000157, loss = 0.002014
grad_step = 000158, loss = 0.002012
grad_step = 000159, loss = 0.002009
grad_step = 000160, loss = 0.002007
grad_step = 000161, loss = 0.002004
grad_step = 000162, loss = 0.002002
grad_step = 000163, loss = 0.001999
grad_step = 000164, loss = 0.001996
grad_step = 000165, loss = 0.001994
grad_step = 000166, loss = 0.001991
grad_step = 000167, loss = 0.001989
grad_step = 000168, loss = 0.001986
grad_step = 000169, loss = 0.001983
grad_step = 000170, loss = 0.001981
grad_step = 000171, loss = 0.001978
grad_step = 000172, loss = 0.001975
grad_step = 000173, loss = 0.001973
grad_step = 000174, loss = 0.001970
grad_step = 000175, loss = 0.001967
grad_step = 000176, loss = 0.001964
grad_step = 000177, loss = 0.001962
grad_step = 000178, loss = 0.001959
grad_step = 000179, loss = 0.001956
grad_step = 000180, loss = 0.001952
grad_step = 000181, loss = 0.001949
grad_step = 000182, loss = 0.001946
grad_step = 000183, loss = 0.001942
grad_step = 000184, loss = 0.001938
grad_step = 000185, loss = 0.001935
grad_step = 000186, loss = 0.001931
grad_step = 000187, loss = 0.001927
grad_step = 000188, loss = 0.001923
grad_step = 000189, loss = 0.001919
grad_step = 000190, loss = 0.001914
grad_step = 000191, loss = 0.001910
grad_step = 000192, loss = 0.001907
grad_step = 000193, loss = 0.001903
grad_step = 000194, loss = 0.001898
grad_step = 000195, loss = 0.001893
grad_step = 000196, loss = 0.001890
grad_step = 000197, loss = 0.001885
grad_step = 000198, loss = 0.001880
grad_step = 000199, loss = 0.001875
grad_step = 000200, loss = 0.001871
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001868
grad_step = 000202, loss = 0.001864
grad_step = 000203, loss = 0.001858
grad_step = 000204, loss = 0.001852
grad_step = 000205, loss = 0.001847
grad_step = 000206, loss = 0.001843
grad_step = 000207, loss = 0.001840
grad_step = 000208, loss = 0.001837
grad_step = 000209, loss = 0.001833
grad_step = 000210, loss = 0.001827
grad_step = 000211, loss = 0.001820
grad_step = 000212, loss = 0.001814
grad_step = 000213, loss = 0.001809
grad_step = 000214, loss = 0.001805
grad_step = 000215, loss = 0.001801
grad_step = 000216, loss = 0.001799
grad_step = 000217, loss = 0.001802
grad_step = 000218, loss = 0.001818
grad_step = 000219, loss = 0.001848
grad_step = 000220, loss = 0.001873
grad_step = 000221, loss = 0.001841
grad_step = 000222, loss = 0.001787
grad_step = 000223, loss = 0.001770
grad_step = 000224, loss = 0.001798
grad_step = 000225, loss = 0.001820
grad_step = 000226, loss = 0.001795
grad_step = 000227, loss = 0.001760
grad_step = 000228, loss = 0.001754
grad_step = 000229, loss = 0.001774
grad_step = 000230, loss = 0.001785
grad_step = 000231, loss = 0.001766
grad_step = 000232, loss = 0.001741
grad_step = 000233, loss = 0.001734
grad_step = 000234, loss = 0.001745
grad_step = 000235, loss = 0.001754
grad_step = 000236, loss = 0.001747
grad_step = 000237, loss = 0.001730
grad_step = 000238, loss = 0.001717
grad_step = 000239, loss = 0.001714
grad_step = 000240, loss = 0.001720
grad_step = 000241, loss = 0.001724
grad_step = 000242, loss = 0.001721
grad_step = 000243, loss = 0.001713
grad_step = 000244, loss = 0.001702
grad_step = 000245, loss = 0.001694
grad_step = 000246, loss = 0.001689
grad_step = 000247, loss = 0.001686
grad_step = 000248, loss = 0.001685
grad_step = 000249, loss = 0.001685
grad_step = 000250, loss = 0.001687
grad_step = 000251, loss = 0.001692
grad_step = 000252, loss = 0.001701
grad_step = 000253, loss = 0.001717
grad_step = 000254, loss = 0.001735
grad_step = 000255, loss = 0.001754
grad_step = 000256, loss = 0.001754
grad_step = 000257, loss = 0.001733
grad_step = 000258, loss = 0.001684
grad_step = 000259, loss = 0.001649
grad_step = 000260, loss = 0.001649
grad_step = 000261, loss = 0.001672
grad_step = 000262, loss = 0.001688
grad_step = 000263, loss = 0.001676
grad_step = 000264, loss = 0.001649
grad_step = 000265, loss = 0.001627
grad_step = 000266, loss = 0.001624
grad_step = 000267, loss = 0.001634
grad_step = 000268, loss = 0.001645
grad_step = 000269, loss = 0.001648
grad_step = 000270, loss = 0.001638
grad_step = 000271, loss = 0.001622
grad_step = 000272, loss = 0.001606
grad_step = 000273, loss = 0.001597
grad_step = 000274, loss = 0.001595
grad_step = 000275, loss = 0.001599
grad_step = 000276, loss = 0.001603
grad_step = 000277, loss = 0.001607
grad_step = 000278, loss = 0.001608
grad_step = 000279, loss = 0.001606
grad_step = 000280, loss = 0.001602
grad_step = 000281, loss = 0.001594
grad_step = 000282, loss = 0.001586
grad_step = 000283, loss = 0.001577
grad_step = 000284, loss = 0.001570
grad_step = 000285, loss = 0.001563
grad_step = 000286, loss = 0.001557
grad_step = 000287, loss = 0.001552
grad_step = 000288, loss = 0.001548
grad_step = 000289, loss = 0.001544
grad_step = 000290, loss = 0.001541
grad_step = 000291, loss = 0.001537
grad_step = 000292, loss = 0.001534
grad_step = 000293, loss = 0.001531
grad_step = 000294, loss = 0.001529
grad_step = 000295, loss = 0.001529
grad_step = 000296, loss = 0.001535
grad_step = 000297, loss = 0.001558
grad_step = 000298, loss = 0.001630
grad_step = 000299, loss = 0.001797
grad_step = 000300, loss = 0.002087
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002107
grad_step = 000302, loss = 0.001764
grad_step = 000303, loss = 0.001519
grad_step = 000304, loss = 0.001772
grad_step = 000305, loss = 0.001871
grad_step = 000306, loss = 0.001560
grad_step = 000307, loss = 0.001612
grad_step = 000308, loss = 0.001799
grad_step = 000309, loss = 0.001585
grad_step = 000310, loss = 0.001558
grad_step = 000311, loss = 0.001699
grad_step = 000312, loss = 0.001565
grad_step = 000313, loss = 0.001525
grad_step = 000314, loss = 0.001640
grad_step = 000315, loss = 0.001535
grad_step = 000316, loss = 0.001512
grad_step = 000317, loss = 0.001599
grad_step = 000318, loss = 0.001524
grad_step = 000319, loss = 0.001495
grad_step = 000320, loss = 0.001564
grad_step = 000321, loss = 0.001510
grad_step = 000322, loss = 0.001481
grad_step = 000323, loss = 0.001533
grad_step = 000324, loss = 0.001498
grad_step = 000325, loss = 0.001472
grad_step = 000326, loss = 0.001506
grad_step = 000327, loss = 0.001489
grad_step = 000328, loss = 0.001464
grad_step = 000329, loss = 0.001486
grad_step = 000330, loss = 0.001478
grad_step = 000331, loss = 0.001459
grad_step = 000332, loss = 0.001469
grad_step = 000333, loss = 0.001467
grad_step = 000334, loss = 0.001453
grad_step = 000335, loss = 0.001457
grad_step = 000336, loss = 0.001456
grad_step = 000337, loss = 0.001447
grad_step = 000338, loss = 0.001449
grad_step = 000339, loss = 0.001446
grad_step = 000340, loss = 0.001439
grad_step = 000341, loss = 0.001441
grad_step = 000342, loss = 0.001439
grad_step = 000343, loss = 0.001431
grad_step = 000344, loss = 0.001431
grad_step = 000345, loss = 0.001432
grad_step = 000346, loss = 0.001426
grad_step = 000347, loss = 0.001422
grad_step = 000348, loss = 0.001423
grad_step = 000349, loss = 0.001421
grad_step = 000350, loss = 0.001415
grad_step = 000351, loss = 0.001414
grad_step = 000352, loss = 0.001413
grad_step = 000353, loss = 0.001410
grad_step = 000354, loss = 0.001407
grad_step = 000355, loss = 0.001405
grad_step = 000356, loss = 0.001403
grad_step = 000357, loss = 0.001400
grad_step = 000358, loss = 0.001398
grad_step = 000359, loss = 0.001396
grad_step = 000360, loss = 0.001394
grad_step = 000361, loss = 0.001391
grad_step = 000362, loss = 0.001389
grad_step = 000363, loss = 0.001387
grad_step = 000364, loss = 0.001385
grad_step = 000365, loss = 0.001382
grad_step = 000366, loss = 0.001380
grad_step = 000367, loss = 0.001378
grad_step = 000368, loss = 0.001375
grad_step = 000369, loss = 0.001373
grad_step = 000370, loss = 0.001371
grad_step = 000371, loss = 0.001369
grad_step = 000372, loss = 0.001366
grad_step = 000373, loss = 0.001364
grad_step = 000374, loss = 0.001361
grad_step = 000375, loss = 0.001359
grad_step = 000376, loss = 0.001357
grad_step = 000377, loss = 0.001354
grad_step = 000378, loss = 0.001352
grad_step = 000379, loss = 0.001350
grad_step = 000380, loss = 0.001348
grad_step = 000381, loss = 0.001346
grad_step = 000382, loss = 0.001345
grad_step = 000383, loss = 0.001346
grad_step = 000384, loss = 0.001350
grad_step = 000385, loss = 0.001358
grad_step = 000386, loss = 0.001376
grad_step = 000387, loss = 0.001409
grad_step = 000388, loss = 0.001441
grad_step = 000389, loss = 0.001479
grad_step = 000390, loss = 0.001445
grad_step = 000391, loss = 0.001394
grad_step = 000392, loss = 0.001353
grad_step = 000393, loss = 0.001350
grad_step = 000394, loss = 0.001359
grad_step = 000395, loss = 0.001350
grad_step = 000396, loss = 0.001349
grad_step = 000397, loss = 0.001356
grad_step = 000398, loss = 0.001360
grad_step = 000399, loss = 0.001355
grad_step = 000400, loss = 0.001327
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001310
grad_step = 000402, loss = 0.001316
grad_step = 000403, loss = 0.001332
grad_step = 000404, loss = 0.001333
grad_step = 000405, loss = 0.001316
grad_step = 000406, loss = 0.001304
grad_step = 000407, loss = 0.001307
grad_step = 000408, loss = 0.001311
grad_step = 000409, loss = 0.001308
grad_step = 000410, loss = 0.001298
grad_step = 000411, loss = 0.001290
grad_step = 000412, loss = 0.001290
grad_step = 000413, loss = 0.001296
grad_step = 000414, loss = 0.001300
grad_step = 000415, loss = 0.001298
grad_step = 000416, loss = 0.001293
grad_step = 000417, loss = 0.001286
grad_step = 000418, loss = 0.001281
grad_step = 000419, loss = 0.001280
grad_step = 000420, loss = 0.001281
grad_step = 000421, loss = 0.001283
grad_step = 000422, loss = 0.001282
grad_step = 000423, loss = 0.001281
grad_step = 000424, loss = 0.001279
grad_step = 000425, loss = 0.001277
grad_step = 000426, loss = 0.001278
grad_step = 000427, loss = 0.001285
grad_step = 000428, loss = 0.001296
grad_step = 000429, loss = 0.001305
grad_step = 000430, loss = 0.001323
grad_step = 000431, loss = 0.001336
grad_step = 000432, loss = 0.001356
grad_step = 000433, loss = 0.001351
grad_step = 000434, loss = 0.001339
grad_step = 000435, loss = 0.001304
grad_step = 000436, loss = 0.001279
grad_step = 000437, loss = 0.001276
grad_step = 000438, loss = 0.001299
grad_step = 000439, loss = 0.001354
grad_step = 000440, loss = 0.001338
grad_step = 000441, loss = 0.001293
grad_step = 000442, loss = 0.001250
grad_step = 000443, loss = 0.001271
grad_step = 000444, loss = 0.001299
grad_step = 000445, loss = 0.001267
grad_step = 000446, loss = 0.001235
grad_step = 000447, loss = 0.001241
grad_step = 000448, loss = 0.001268
grad_step = 000449, loss = 0.001284
grad_step = 000450, loss = 0.001258
grad_step = 000451, loss = 0.001231
grad_step = 000452, loss = 0.001223
grad_step = 000453, loss = 0.001236
grad_step = 000454, loss = 0.001248
grad_step = 000455, loss = 0.001237
grad_step = 000456, loss = 0.001220
grad_step = 000457, loss = 0.001212
grad_step = 000458, loss = 0.001219
grad_step = 000459, loss = 0.001227
grad_step = 000460, loss = 0.001222
grad_step = 000461, loss = 0.001212
grad_step = 000462, loss = 0.001205
grad_step = 000463, loss = 0.001207
grad_step = 000464, loss = 0.001212
grad_step = 000465, loss = 0.001214
grad_step = 000466, loss = 0.001212
grad_step = 000467, loss = 0.001209
grad_step = 000468, loss = 0.001214
grad_step = 000469, loss = 0.001232
grad_step = 000470, loss = 0.001262
grad_step = 000471, loss = 0.001318
grad_step = 000472, loss = 0.001365
grad_step = 000473, loss = 0.001417
grad_step = 000474, loss = 0.001370
grad_step = 000475, loss = 0.001282
grad_step = 000476, loss = 0.001206
grad_step = 000477, loss = 0.001211
grad_step = 000478, loss = 0.001266
grad_step = 000479, loss = 0.001258
grad_step = 000480, loss = 0.001203
grad_step = 000481, loss = 0.001179
grad_step = 000482, loss = 0.001212
grad_step = 000483, loss = 0.001227
grad_step = 000484, loss = 0.001192
grad_step = 000485, loss = 0.001173
grad_step = 000486, loss = 0.001186
grad_step = 000487, loss = 0.001205
grad_step = 000488, loss = 0.001201
grad_step = 000489, loss = 0.001178
grad_step = 000490, loss = 0.001160
grad_step = 000491, loss = 0.001158
grad_step = 000492, loss = 0.001169
grad_step = 000493, loss = 0.001182
grad_step = 000494, loss = 0.001187
grad_step = 000495, loss = 0.001187
grad_step = 000496, loss = 0.001189
grad_step = 000497, loss = 0.001198
grad_step = 000498, loss = 0.001232
grad_step = 000499, loss = 0.001239
grad_step = 000500, loss = 0.001237
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001179
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
  (date_run                              2020-05-09 04:54:06.234631
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.228622
metric_name                                  mean_absolute_error
Name: 0, dtype: object,) 
  (date_run                              2020-05-09 04:54:06.241431
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.123266
metric_name                                   mean_squared_error
Name: 1, dtype: object,) 
  (date_run                              2020-05-09 04:54:06.250010
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  0.137059
metric_name                                median_absolute_error
Name: 2, dtype: object,) 
  (date_run                              2020-05-09 04:54:06.256369
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                 -0.873065
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f3a381863c8> <class 'mlmodels.model_keras.armdn.Model'>
  ('#### Loading dataset   #############################################',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355457.8438
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 262776.8750
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 168163.7031
Epoch 4/10

1/1 [==============================] - 0s 117ms/step - loss: 94846.7656
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 51265.1953
Epoch 6/10

1/1 [==============================] - 0s 106ms/step - loss: 29182.6230
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 18015.1035
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 12151.1758
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 8621.2119
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 6558.1489
  ('#### Inference Need return ypred, ytrue #########################',) 
[[ 3.90969813e-01  1.01700163e+01  7.98212147e+00  1.03736248e+01
   1.01280012e+01  7.68655252e+00  8.26419830e+00  9.61921406e+00
   7.54412651e+00  9.30254841e+00  8.07492352e+00  7.02465105e+00
   8.27923012e+00  7.65906048e+00  6.48113728e+00  8.37891579e+00
   7.78621054e+00  8.13146496e+00  7.82631254e+00  9.17081451e+00
   7.39859152e+00  6.79287910e+00  8.03358364e+00  8.05028725e+00
   7.67507029e+00  9.26933575e+00  8.29851437e+00  9.29163933e+00
   8.83367634e+00  7.76882553e+00  9.52826118e+00  6.29998493e+00
   6.62061357e+00  8.37103939e+00  8.33783913e+00  8.41376019e+00
   9.26635170e+00  7.15787125e+00  8.72294903e+00  8.08781910e+00
   9.32544422e+00  9.17663479e+00  9.40505314e+00  9.02359295e+00
   6.79281330e+00  7.03596687e+00  8.26144314e+00  9.09197712e+00
   8.96441841e+00  7.97502899e+00  1.00822220e+01  8.19250298e+00
   8.18217945e+00  9.10270500e+00  8.66751671e+00  7.03080177e+00
   7.26632786e+00  8.39556694e+00  8.00595760e+00  9.38710594e+00
   4.55826133e-01  1.55625033e+00 -1.11300039e+00  6.17540002e-01
  -1.46037614e+00  1.51912963e+00  5.55470526e-01  1.30813706e+00
  -1.55640811e-01 -9.50502276e-01  8.99786890e-01  6.90723896e-01
   1.58784166e-01  1.65755713e+00 -6.83905542e-01 -6.50116622e-01
  -1.40198362e+00 -1.26353467e+00 -1.00183737e+00 -1.56858218e+00
   5.08491278e-01 -2.01238203e+00 -4.22062427e-01 -5.81494451e-01
  -1.12010241e+00  4.02794369e-02 -1.14511216e+00 -5.79142809e-01
  -4.81936336e-02  1.30518377e+00 -1.65841162e-01  1.25392032e+00
   6.62445307e-01  1.40307558e+00 -6.51482642e-01 -1.69983006e+00
   5.59343696e-01 -3.90971363e-01  1.10358667e+00 -4.14576113e-01
   3.74160051e-01  9.80841994e-01 -1.52547181e-01 -2.52452165e-01
   3.93302619e-01  1.97736353e-01 -1.04147315e+00  8.26813340e-01
  -8.69761407e-01 -4.12591279e-01 -1.77223712e-01  4.24329549e-01
  -3.00440460e-01  1.88303173e+00 -1.01783383e+00 -1.78314078e+00
  -4.77664024e-02 -1.69421661e+00  3.04087400e-01 -2.70869553e-01
  -3.19225967e-01  1.49154878e+00 -3.43678385e-01 -1.64437801e-01
  -3.49582136e-01 -1.46009946e+00 -6.30897880e-01  3.91870022e-01
  -2.72628397e-01 -1.19256592e+00 -6.03598356e-03  1.97060633e+00
  -5.43486536e-01  1.01211464e+00  6.01324558e-01 -1.39554453e+00
   1.00841773e+00 -6.11131430e-01  6.33466601e-01  6.73307776e-01
  -7.41059244e-01 -2.01866293e+00  3.89327288e-01 -8.27528834e-01
  -1.86699104e+00  8.76059353e-01 -2.44792625e-01  5.81564978e-02
  -5.67837358e-01 -1.28043365e+00 -1.49272299e+00 -1.06736803e+00
  -5.03881991e-01 -3.84573899e-02 -2.96766400e-01  2.24429667e-01
   5.15156627e-01  7.06481040e-02  4.14934516e-01 -1.40006137e+00
   6.30271137e-01  8.77356589e-01 -4.00581926e-01  3.86817634e-01
   1.19931340e+00 -8.33961546e-01 -1.70298874e-01 -1.19653189e+00
   1.21603918e+00 -6.84795380e-01  1.31538224e+00 -2.41998196e+00
   5.16973436e-01  6.66629076e-01  2.14179111e+00  4.12449360e-01
  -3.67955059e-01  2.81543545e-02  2.03234076e+00  1.15271664e+00
   4.01233435e-02  8.75474548e+00  9.21735191e+00  1.02675724e+01
   1.01329393e+01  8.44368744e+00  7.80240631e+00  8.65825939e+00
   8.72514915e+00  9.64474678e+00  8.76588726e+00  9.27682304e+00
   8.92833424e+00  1.01256485e+01  9.20272350e+00  7.90085840e+00
   8.96407223e+00  7.79809618e+00  8.96691036e+00  8.69498062e+00
   9.23880291e+00  9.09035683e+00  1.02468500e+01  8.68570518e+00
   6.90020037e+00  9.48050499e+00  8.73336315e+00  9.66623974e+00
   6.98190832e+00  7.96628046e+00  7.83069801e+00  6.92030811e+00
   7.24677324e+00  7.38354874e+00  9.49394798e+00  8.04094791e+00
   8.99816418e+00  8.84923077e+00  9.49083424e+00  8.56453323e+00
   7.71420908e+00  7.36716700e+00  8.28025246e+00  8.84621620e+00
   7.17242050e+00  8.73242760e+00  8.12032032e+00  9.63301563e+00
   7.83239794e+00  8.57921791e+00  1.00468397e+01  8.83979321e+00
   7.46404934e+00  9.30476093e+00  7.78147364e+00  8.91677284e+00
   6.77716351e+00  6.80972672e+00  8.34215832e+00  8.91485882e+00
   5.13515115e-01  1.11281025e+00  3.12910795e+00  1.10798037e+00
   7.87115395e-01  8.31419945e-01  4.66049135e-01  5.36308527e-01
   9.36783135e-01  2.53880167e+00  3.00768757e+00  1.85432875e+00
   2.42678881e+00  1.46698546e+00  4.82636869e-01  1.05341399e+00
   5.25334775e-01  6.43600345e-01  1.33284891e+00  6.13453329e-01
   2.52548480e+00  1.84616196e+00  3.11983538e+00  1.43235564e+00
   3.13984036e-01  2.39126265e-01  1.20993471e+00  5.78544080e-01
   4.09364522e-01  5.59064865e-01  2.59094763e+00  5.91801405e-01
   1.08885002e+00  1.52884912e+00  1.42046511e-01  2.46495962e+00
   1.26421380e+00  1.55867982e+00  9.43533063e-01  3.24321628e-01
   6.22853041e-01  2.00937033e+00  6.43388331e-01  8.38523626e-01
   8.32655132e-01  2.69354820e-01  1.19764018e+00  9.29496944e-01
   3.38942230e-01  1.22874117e+00  1.59497702e+00  1.90385127e+00
   9.12893176e-01  2.19627285e+00  5.56889653e-01  1.00259805e+00
   1.23090565e+00  1.11974692e+00  2.53257656e+00  1.43068790e-01
   3.02235961e-01  8.59470963e-01  1.06279516e+00  5.49191236e-01
   2.05357313e+00  1.47815454e+00  1.48752999e+00  6.91840827e-01
   4.15850520e-01  8.09663475e-01  1.85224509e+00  2.60080814e-01
   2.62371397e+00  1.66976726e+00  1.33873999e+00  8.83388102e-01
   1.27999496e+00  3.10346270e+00  2.07110310e+00  8.18833232e-01
   2.42715538e-01  7.02464104e-01  6.50240421e-01  8.49530935e-01
   2.46219015e+00  2.60355282e+00  1.78946984e+00  1.92255723e+00
   2.82138646e-01  8.71173918e-01  1.16293335e+00  3.74010742e-01
   2.01458693e+00  5.30710816e-01  1.85804069e+00  2.15196061e+00
   1.69242001e+00  9.93195534e-01  5.13421953e-01  3.01059067e-01
   9.70847309e-01  2.65398502e+00  8.86656582e-01  1.26291645e+00
   3.24040413e-01  3.79387736e-01  2.13260174e-01  5.98618805e-01
   1.25504899e+00  1.06821871e+00  5.00449359e-01  9.62014258e-01
   5.62920630e-01  6.84949875e-01  1.39394569e+00  6.16753697e-01
   1.36977363e+00  1.16639352e+00  5.25279164e-01  5.15452743e-01
   7.37340927e+00 -1.19863534e+00 -6.66343212e+00]]
  ('### Calculate Metrics    ########################################',) 
  (date_run                              2020-05-09 04:54:15.284710
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.0738
metric_name                                  mean_absolute_error
Name: 4, dtype: object,) 
  (date_run                              2020-05-09 04:54:15.289121
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   8871.98
metric_name                                   mean_squared_error
Name: 5, dtype: object,) 
  (date_run                              2020-05-09 04:54:15.293196
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                   94.3929
metric_name                                median_absolute_error
Name: 6, dtype: object,) 
  (date_run                              2020-05-09 04:54:15.296551
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
metric                                                  -793.549
metric_name                                             r2_score
Name: 7, dtype: object,) 
  ("### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_uri': 'model_gluon/fb_prophet.py'},) 
  ('#### Setup Model   ##############################################',) 
  ('#### Fit  #######################################################',) 
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f3a31de1828> <class 'mlmodels.model_gluon.fb_prophet.Model'>
  ({'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close', 'train': True}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, FileNotFoundError(2, "File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist")) 
  ("### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
  ('#### Model URI and Config JSON',) 
  ({'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}},) 
  ('#### Setup Model   ##############################################',) 
  ({'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, KeyError('model_uri',)) 
  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/',) 
  (                     date_run  ...            metric_name
0  2020-05-09 04:54:06.234631  ...    mean_absolute_error
1  2020-05-09 04:54:06.241431  ...     mean_squared_error
2  2020-05-09 04:54:06.250010  ...  median_absolute_error
3  2020-05-09 04:54:06.256369  ...               r2_score
4  2020-05-09 04:54:15.284710  ...    mean_absolute_error
5  2020-05-09 04:54:15.289121  ...     mean_squared_error
6  2020-05-09 04:54:15.293196  ...  median_absolute_error
7  2020-05-09 04:54:15.296551  ...               r2_score

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
ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt

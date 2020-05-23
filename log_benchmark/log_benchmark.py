
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '7423a9c1aea8d708841a3941e104542978e088ce', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7423a9c1aea8d708841a3941e104542978e088ce

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  timeseries 

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7d04748eb8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-23 00:30:41.644503
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-23 00:30:41.648080
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-23 00:30:41.650833
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-23 00:30:41.653478
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7d105132e8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354445.7812
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 264233.2500
Epoch 3/10

1/1 [==============================] - 0s 86ms/step - loss: 150783.5469
Epoch 4/10

1/1 [==============================] - 0s 89ms/step - loss: 78577.7500
Epoch 5/10

1/1 [==============================] - 0s 84ms/step - loss: 43244.7227
Epoch 6/10

1/1 [==============================] - 0s 81ms/step - loss: 25990.8906
Epoch 7/10

1/1 [==============================] - 0s 83ms/step - loss: 16993.9766
Epoch 8/10

1/1 [==============================] - 0s 86ms/step - loss: 11913.4424
Epoch 9/10

1/1 [==============================] - 0s 84ms/step - loss: 8855.2637
Epoch 10/10

1/1 [==============================] - 0s 80ms/step - loss: 6908.4380

  #### Inference Need return ypred, ytrue ######################### 
[[-0.10534086  9.37565     7.3757086   8.57574     7.0719028   8.297211
   8.969006    7.087898    8.564823    7.4898977   8.222       7.231206
   7.9483204   7.7051854   7.9671187   8.232798    7.5649266   6.9377155
   8.412699    8.977525    7.639578    7.991881    8.000341    6.927081
   8.220136    7.9448285   7.7857437   8.414316    7.908832    8.429702
   8.227682    8.226904    8.134851    7.6974797   6.563505    8.905971
   6.99009     6.1073623   7.2516727   6.6889124   8.048792    7.816808
   8.211421    7.2031693   7.089614    8.522499    7.0994835   7.9301023
   7.5788674   8.362539    7.6189218   8.9310465   7.158923    7.3459487
   8.160761    7.4624586   7.971987    7.706856    6.7579627   7.820224
   0.3011452  -0.52978384 -0.08077431  0.61558324 -0.5320395  -0.8755394
   0.7589058   0.22800502  0.23806882 -0.74566436  0.43889046  1.0028784
   0.2833576  -0.9603274   1.4021822   0.538807   -0.9953848   1.158444
   0.2703218   0.44491893 -0.17697585 -0.39089686 -0.6607803   0.44589764
  -0.61666954  1.3311752   0.36070216 -0.72015953  1.5331559  -0.02108371
  -0.55979186 -0.6625489   0.6869905   1.3708389  -0.27266797  1.6042253
   0.01743083  0.88250357  0.6935429   0.3258816  -0.7562109   1.34337
  -1.1530058   0.9252626  -1.3892499   0.32376504 -0.47718835 -0.01669744
   0.08852957 -0.13571532 -0.39726844 -1.1578772  -0.1436603  -0.83175105
   0.13155243 -0.5070148   0.5464493  -0.2559019  -0.07224619  1.0090444
   0.8720387  -0.34439275  0.20133147  0.11762245 -0.9215746   0.9590901
  -1.3773379  -0.61717683 -0.6314763  -0.01838776  0.2510137  -0.08965549
   0.87870693  0.5682751   0.6984862  -1.2521844  -0.19677027  0.8023801
  -0.59059304  0.03071913 -0.81148803 -0.5860344   0.8399364   0.19299202
   0.8809395   0.14474201 -0.03448653  0.5064203  -0.9954747   0.20728377
  -0.11890362  0.2987714   0.79765296 -0.77270985  0.49337727  0.25295657
  -0.29666963 -0.8485626   0.8943202   0.6901644   1.0086527  -0.33772993
   0.9783588   0.65130484  1.3927251  -0.27528417 -0.77297425 -1.6231139
   0.60422623 -0.06939296 -0.37016574 -0.2873907   0.10992181  0.67296183
   1.444518    0.03857189  0.4514435  -0.32028407  0.01654518 -0.8688185
   0.16546977  7.972693    7.407169    7.803173    8.065381    8.128026
   8.64573     8.24375     8.096824    6.994504    8.070584    8.637893
   7.3920784   8.172301    7.706187    8.341146    7.544746    6.860064
   8.018503    8.49051     7.2610664   6.8769355   8.089755    8.496736
   7.3538785   8.381848    8.393639    8.606484    7.9515386   7.86866
   8.471464    8.115364    8.4102      8.055805    9.056134    8.472472
   7.617527    8.446129    9.046442    9.140688    7.630737    8.811069
   9.443783    8.554061    8.053349    8.635127    7.628       8.369119
   8.085098    9.104639    8.191457    8.192556    7.4545417   7.0124283
   7.902741    7.5617127   8.048446    7.757574    7.421163    8.737258
   1.3046639   0.9362141   1.4154708   1.2141123   1.1988581   0.32177436
   0.87955695  0.33780408  1.9781265   0.9555583   1.0679747   1.3284925
   1.5218565   0.5815879   1.4784942   0.67394334  2.5464625   0.54344505
   0.3160758   2.561071    1.9464855   0.8971997   0.4380052   1.4395013
   1.4290494   0.94612145  0.56001747  1.0929908   1.0698861   0.36533594
   1.5884233   0.47614807  1.3682115   1.3714066   1.4705309   2.6331124
   1.166794    0.86275065  0.71053153  1.3598949   1.6987679   1.0233256
   1.8533121   1.363438    1.1566743   0.8982223   0.8907796   1.5173626
   1.4562436   1.3595697   1.5032517   1.0981476   1.246542    1.3223214
   1.8746041   0.87129533  1.2825446   0.58904296  0.55413115  0.3515383
   1.0974779   1.3335166   1.2856588   0.6100274   0.67213005  0.8372437
   0.63369393  0.77164227  1.7596984   2.3447113   1.3643148   0.6643858
   0.81861657  0.45707136  1.5748957   1.9540625   0.8986933   1.369299
   2.4312267   0.55944693  1.3444152   1.9044384   0.5175979   1.0359483
   0.79303557  0.8288897   1.0631651   1.2970839   0.295353    1.9296739
   0.63892305  0.75765264  2.4109511   1.1290201   0.46098733  2.508178
   1.1556246   0.36421293  1.5615835   0.24075079  0.5244739   0.9167923
   1.7253908   1.2463632   1.4128911   1.051616    0.6392213   0.3356433
   0.3545195   1.464976    1.1535003   0.70971096  2.0793338   0.6352683
   0.33943278  1.2333503   0.67844963  0.3234663   0.39032376  1.3075751
   6.374082   -8.56247    -4.749426  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-23 00:30:52.988779
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.304
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-23 00:30:52.992127
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8912.07
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-23 00:30:52.994673
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.9158
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-23 00:30:52.997805
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -797.139
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140174580126216
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140172050862656
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140172050863160
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140172050863664
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140172050864168
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140172050864672

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7d04408b00> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.495281
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.464149
grad_step = 000002, loss = 0.444056
grad_step = 000003, loss = 0.424038
grad_step = 000004, loss = 0.402581
grad_step = 000005, loss = 0.383265
grad_step = 000006, loss = 0.369188
grad_step = 000007, loss = 0.360029
grad_step = 000008, loss = 0.349207
grad_step = 000009, loss = 0.334774
grad_step = 000010, loss = 0.322368
grad_step = 000011, loss = 0.312946
grad_step = 000012, loss = 0.304089
grad_step = 000013, loss = 0.294516
grad_step = 000014, loss = 0.284053
grad_step = 000015, loss = 0.273267
grad_step = 000016, loss = 0.262613
grad_step = 000017, loss = 0.252683
grad_step = 000018, loss = 0.243660
grad_step = 000019, loss = 0.234903
grad_step = 000020, loss = 0.225756
grad_step = 000021, loss = 0.216502
grad_step = 000022, loss = 0.207823
grad_step = 000023, loss = 0.199708
grad_step = 000024, loss = 0.191609
grad_step = 000025, loss = 0.183290
grad_step = 000026, loss = 0.174867
grad_step = 000027, loss = 0.166652
grad_step = 000028, loss = 0.159023
grad_step = 000029, loss = 0.151949
grad_step = 000030, loss = 0.144955
grad_step = 000031, loss = 0.137860
grad_step = 000032, loss = 0.130897
grad_step = 000033, loss = 0.124266
grad_step = 000034, loss = 0.117917
grad_step = 000035, loss = 0.111691
grad_step = 000036, loss = 0.105625
grad_step = 000037, loss = 0.099845
grad_step = 000038, loss = 0.094348
grad_step = 000039, loss = 0.089019
grad_step = 000040, loss = 0.083784
grad_step = 000041, loss = 0.078811
grad_step = 000042, loss = 0.074178
grad_step = 000043, loss = 0.069743
grad_step = 000044, loss = 0.065392
grad_step = 000045, loss = 0.061189
grad_step = 000046, loss = 0.057293
grad_step = 000047, loss = 0.053666
grad_step = 000048, loss = 0.050193
grad_step = 000049, loss = 0.046862
grad_step = 000050, loss = 0.043736
grad_step = 000051, loss = 0.040807
grad_step = 000052, loss = 0.038016
grad_step = 000053, loss = 0.035389
grad_step = 000054, loss = 0.032974
grad_step = 000055, loss = 0.030735
grad_step = 000056, loss = 0.028589
grad_step = 000057, loss = 0.026588
grad_step = 000058, loss = 0.024798
grad_step = 000059, loss = 0.023310
grad_step = 000060, loss = 0.021979
grad_step = 000061, loss = 0.020418
grad_step = 000062, loss = 0.018519
grad_step = 000063, loss = 0.017399
grad_step = 000064, loss = 0.016489
grad_step = 000065, loss = 0.014986
grad_step = 000066, loss = 0.014009
grad_step = 000067, loss = 0.013286
grad_step = 000068, loss = 0.012104
grad_step = 000069, loss = 0.011382
grad_step = 000070, loss = 0.010763
grad_step = 000071, loss = 0.009841
grad_step = 000072, loss = 0.009327
grad_step = 000073, loss = 0.008779
grad_step = 000074, loss = 0.008063
grad_step = 000075, loss = 0.007703
grad_step = 000076, loss = 0.007231
grad_step = 000077, loss = 0.006683
grad_step = 000078, loss = 0.006409
grad_step = 000079, loss = 0.006013
grad_step = 000080, loss = 0.005601
grad_step = 000081, loss = 0.005395
grad_step = 000082, loss = 0.005075
grad_step = 000083, loss = 0.004767
grad_step = 000084, loss = 0.004612
grad_step = 000085, loss = 0.004365
grad_step = 000086, loss = 0.004133
grad_step = 000087, loss = 0.004017
grad_step = 000088, loss = 0.003835
grad_step = 000089, loss = 0.003654
grad_step = 000090, loss = 0.003569
grad_step = 000091, loss = 0.003445
grad_step = 000092, loss = 0.003301
grad_step = 000093, loss = 0.003232
grad_step = 000094, loss = 0.003156
grad_step = 000095, loss = 0.003043
grad_step = 000096, loss = 0.002977
grad_step = 000097, loss = 0.002933
grad_step = 000098, loss = 0.002854
grad_step = 000099, loss = 0.002789
grad_step = 000100, loss = 0.002757
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002712
grad_step = 000102, loss = 0.002654
grad_step = 000103, loss = 0.002618
grad_step = 000104, loss = 0.002595
grad_step = 000105, loss = 0.002559
grad_step = 000106, loss = 0.002518
grad_step = 000107, loss = 0.002494
grad_step = 000108, loss = 0.002478
grad_step = 000109, loss = 0.002453
grad_step = 000110, loss = 0.002424
grad_step = 000111, loss = 0.002404
grad_step = 000112, loss = 0.002392
grad_step = 000113, loss = 0.002378
grad_step = 000114, loss = 0.002360
grad_step = 000115, loss = 0.002342
grad_step = 000116, loss = 0.002331
grad_step = 000117, loss = 0.002323
grad_step = 000118, loss = 0.002315
grad_step = 000119, loss = 0.002304
grad_step = 000120, loss = 0.002292
grad_step = 000121, loss = 0.002284
grad_step = 000122, loss = 0.002278
grad_step = 000123, loss = 0.002274
grad_step = 000124, loss = 0.002270
grad_step = 000125, loss = 0.002265
grad_step = 000126, loss = 0.002259
grad_step = 000127, loss = 0.002254
grad_step = 000128, loss = 0.002249
grad_step = 000129, loss = 0.002244
grad_step = 000130, loss = 0.002241
grad_step = 000131, loss = 0.002238
grad_step = 000132, loss = 0.002236
grad_step = 000133, loss = 0.002234
grad_step = 000134, loss = 0.002233
grad_step = 000135, loss = 0.002235
grad_step = 000136, loss = 0.002242
grad_step = 000137, loss = 0.002259
grad_step = 000138, loss = 0.002301
grad_step = 000139, loss = 0.002390
grad_step = 000140, loss = 0.002566
grad_step = 000141, loss = 0.002792
grad_step = 000142, loss = 0.002946
grad_step = 000143, loss = 0.002691
grad_step = 000144, loss = 0.002305
grad_step = 000145, loss = 0.002247
grad_step = 000146, loss = 0.002507
grad_step = 000147, loss = 0.002621
grad_step = 000148, loss = 0.002368
grad_step = 000149, loss = 0.002211
grad_step = 000150, loss = 0.002367
grad_step = 000151, loss = 0.002468
grad_step = 000152, loss = 0.002322
grad_step = 000153, loss = 0.002204
grad_step = 000154, loss = 0.002309
grad_step = 000155, loss = 0.002380
grad_step = 000156, loss = 0.002269
grad_step = 000157, loss = 0.002203
grad_step = 000158, loss = 0.002280
grad_step = 000159, loss = 0.002313
grad_step = 000160, loss = 0.002237
grad_step = 000161, loss = 0.002199
grad_step = 000162, loss = 0.002250
grad_step = 000163, loss = 0.002272
grad_step = 000164, loss = 0.002222
grad_step = 000165, loss = 0.002194
grad_step = 000166, loss = 0.002224
grad_step = 000167, loss = 0.002242
grad_step = 000168, loss = 0.002212
grad_step = 000169, loss = 0.002188
grad_step = 000170, loss = 0.002201
grad_step = 000171, loss = 0.002219
grad_step = 000172, loss = 0.002208
grad_step = 000173, loss = 0.002185
grad_step = 000174, loss = 0.002183
grad_step = 000175, loss = 0.002199
grad_step = 000176, loss = 0.002201
grad_step = 000177, loss = 0.002186
grad_step = 000178, loss = 0.002174
grad_step = 000179, loss = 0.002180
grad_step = 000180, loss = 0.002189
grad_step = 000181, loss = 0.002186
grad_step = 000182, loss = 0.002175
grad_step = 000183, loss = 0.002169
grad_step = 000184, loss = 0.002171
grad_step = 000185, loss = 0.002176
grad_step = 000186, loss = 0.002177
grad_step = 000187, loss = 0.002171
grad_step = 000188, loss = 0.002165
grad_step = 000189, loss = 0.002162
grad_step = 000190, loss = 0.002163
grad_step = 000191, loss = 0.002165
grad_step = 000192, loss = 0.002166
grad_step = 000193, loss = 0.002165
grad_step = 000194, loss = 0.002161
grad_step = 000195, loss = 0.002158
grad_step = 000196, loss = 0.002155
grad_step = 000197, loss = 0.002153
grad_step = 000198, loss = 0.002153
grad_step = 000199, loss = 0.002153
grad_step = 000200, loss = 0.002153
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002154
grad_step = 000202, loss = 0.002155
grad_step = 000203, loss = 0.002157
grad_step = 000204, loss = 0.002160
grad_step = 000205, loss = 0.002165
grad_step = 000206, loss = 0.002174
grad_step = 000207, loss = 0.002192
grad_step = 000208, loss = 0.002221
grad_step = 000209, loss = 0.002271
grad_step = 000210, loss = 0.002326
grad_step = 000211, loss = 0.002395
grad_step = 000212, loss = 0.002430
grad_step = 000213, loss = 0.002417
grad_step = 000214, loss = 0.002341
grad_step = 000215, loss = 0.002233
grad_step = 000216, loss = 0.002157
grad_step = 000217, loss = 0.002144
grad_step = 000218, loss = 0.002187
grad_step = 000219, loss = 0.002250
grad_step = 000220, loss = 0.002282
grad_step = 000221, loss = 0.002268
grad_step = 000222, loss = 0.002215
grad_step = 000223, loss = 0.002163
grad_step = 000224, loss = 0.002139
grad_step = 000225, loss = 0.002143
grad_step = 000226, loss = 0.002164
grad_step = 000227, loss = 0.002187
grad_step = 000228, loss = 0.002200
grad_step = 000229, loss = 0.002197
grad_step = 000230, loss = 0.002173
grad_step = 000231, loss = 0.002146
grad_step = 000232, loss = 0.002128
grad_step = 000233, loss = 0.002127
grad_step = 000234, loss = 0.002137
grad_step = 000235, loss = 0.002149
grad_step = 000236, loss = 0.002156
grad_step = 000237, loss = 0.002157
grad_step = 000238, loss = 0.002154
grad_step = 000239, loss = 0.002148
grad_step = 000240, loss = 0.002139
grad_step = 000241, loss = 0.002128
grad_step = 000242, loss = 0.002121
grad_step = 000243, loss = 0.002118
grad_step = 000244, loss = 0.002118
grad_step = 000245, loss = 0.002121
grad_step = 000246, loss = 0.002123
grad_step = 000247, loss = 0.002126
grad_step = 000248, loss = 0.002130
grad_step = 000249, loss = 0.002136
grad_step = 000250, loss = 0.002145
grad_step = 000251, loss = 0.002160
grad_step = 000252, loss = 0.002180
grad_step = 000253, loss = 0.002210
grad_step = 000254, loss = 0.002249
grad_step = 000255, loss = 0.002302
grad_step = 000256, loss = 0.002362
grad_step = 000257, loss = 0.002413
grad_step = 000258, loss = 0.002434
grad_step = 000259, loss = 0.002381
grad_step = 000260, loss = 0.002284
grad_step = 000261, loss = 0.002171
grad_step = 000262, loss = 0.002111
grad_step = 000263, loss = 0.002120
grad_step = 000264, loss = 0.002175
grad_step = 000265, loss = 0.002231
grad_step = 000266, loss = 0.002243
grad_step = 000267, loss = 0.002211
grad_step = 000268, loss = 0.002152
grad_step = 000269, loss = 0.002109
grad_step = 000270, loss = 0.002103
grad_step = 000271, loss = 0.002128
grad_step = 000272, loss = 0.002160
grad_step = 000273, loss = 0.002175
grad_step = 000274, loss = 0.002167
grad_step = 000275, loss = 0.002142
grad_step = 000276, loss = 0.002115
grad_step = 000277, loss = 0.002101
grad_step = 000278, loss = 0.002101
grad_step = 000279, loss = 0.002111
grad_step = 000280, loss = 0.002121
grad_step = 000281, loss = 0.002127
grad_step = 000282, loss = 0.002125
grad_step = 000283, loss = 0.002118
grad_step = 000284, loss = 0.002112
grad_step = 000285, loss = 0.002107
grad_step = 000286, loss = 0.002107
grad_step = 000287, loss = 0.002106
grad_step = 000288, loss = 0.002105
grad_step = 000289, loss = 0.002101
grad_step = 000290, loss = 0.002096
grad_step = 000291, loss = 0.002091
grad_step = 000292, loss = 0.002089
grad_step = 000293, loss = 0.002090
grad_step = 000294, loss = 0.002094
grad_step = 000295, loss = 0.002099
grad_step = 000296, loss = 0.002103
grad_step = 000297, loss = 0.002108
grad_step = 000298, loss = 0.002110
grad_step = 000299, loss = 0.002113
grad_step = 000300, loss = 0.002115
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002124
grad_step = 000302, loss = 0.002139
grad_step = 000303, loss = 0.002164
grad_step = 000304, loss = 0.002204
grad_step = 000305, loss = 0.002254
grad_step = 000306, loss = 0.002323
grad_step = 000307, loss = 0.002392
grad_step = 000308, loss = 0.002457
grad_step = 000309, loss = 0.002469
grad_step = 000310, loss = 0.002408
grad_step = 000311, loss = 0.002283
grad_step = 000312, loss = 0.002149
grad_step = 000313, loss = 0.002079
grad_step = 000314, loss = 0.002096
grad_step = 000315, loss = 0.002167
grad_step = 000316, loss = 0.002228
grad_step = 000317, loss = 0.002233
grad_step = 000318, loss = 0.002180
grad_step = 000319, loss = 0.002110
grad_step = 000320, loss = 0.002072
grad_step = 000321, loss = 0.002081
grad_step = 000322, loss = 0.002118
grad_step = 000323, loss = 0.002149
grad_step = 000324, loss = 0.002148
grad_step = 000325, loss = 0.002119
grad_step = 000326, loss = 0.002086
grad_step = 000327, loss = 0.002068
grad_step = 000328, loss = 0.002073
grad_step = 000329, loss = 0.002090
grad_step = 000330, loss = 0.002104
grad_step = 000331, loss = 0.002105
grad_step = 000332, loss = 0.002094
grad_step = 000333, loss = 0.002079
grad_step = 000334, loss = 0.002067
grad_step = 000335, loss = 0.002063
grad_step = 000336, loss = 0.002067
grad_step = 000337, loss = 0.002073
grad_step = 000338, loss = 0.002077
grad_step = 000339, loss = 0.002077
grad_step = 000340, loss = 0.002071
grad_step = 000341, loss = 0.002067
grad_step = 000342, loss = 0.002064
grad_step = 000343, loss = 0.002061
grad_step = 000344, loss = 0.002058
grad_step = 000345, loss = 0.002056
grad_step = 000346, loss = 0.002054
grad_step = 000347, loss = 0.002053
grad_step = 000348, loss = 0.002052
grad_step = 000349, loss = 0.002051
grad_step = 000350, loss = 0.002050
grad_step = 000351, loss = 0.002050
grad_step = 000352, loss = 0.002050
grad_step = 000353, loss = 0.002051
grad_step = 000354, loss = 0.002053
grad_step = 000355, loss = 0.002057
grad_step = 000356, loss = 0.002066
grad_step = 000357, loss = 0.002078
grad_step = 000358, loss = 0.002100
grad_step = 000359, loss = 0.002120
grad_step = 000360, loss = 0.002147
grad_step = 000361, loss = 0.002147
grad_step = 000362, loss = 0.002149
grad_step = 000363, loss = 0.002151
grad_step = 000364, loss = 0.002195
grad_step = 000365, loss = 0.002306
grad_step = 000366, loss = 0.002448
grad_step = 000367, loss = 0.002661
grad_step = 000368, loss = 0.002755
grad_step = 000369, loss = 0.002776
grad_step = 000370, loss = 0.002570
grad_step = 000371, loss = 0.002304
grad_step = 000372, loss = 0.002144
grad_step = 000373, loss = 0.002109
grad_step = 000374, loss = 0.002245
grad_step = 000375, loss = 0.002388
grad_step = 000376, loss = 0.002283
grad_step = 000377, loss = 0.002090
grad_step = 000378, loss = 0.002053
grad_step = 000379, loss = 0.002153
grad_step = 000380, loss = 0.002211
grad_step = 000381, loss = 0.002165
grad_step = 000382, loss = 0.002096
grad_step = 000383, loss = 0.002058
grad_step = 000384, loss = 0.002076
grad_step = 000385, loss = 0.002135
grad_step = 000386, loss = 0.002138
grad_step = 000387, loss = 0.002063
grad_step = 000388, loss = 0.002032
grad_step = 000389, loss = 0.002066
grad_step = 000390, loss = 0.002086
grad_step = 000391, loss = 0.002078
grad_step = 000392, loss = 0.002066
grad_step = 000393, loss = 0.002047
grad_step = 000394, loss = 0.002029
grad_step = 000395, loss = 0.002037
grad_step = 000396, loss = 0.002059
grad_step = 000397, loss = 0.002061
grad_step = 000398, loss = 0.002036
grad_step = 000399, loss = 0.002024
grad_step = 000400, loss = 0.002027
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.002030
grad_step = 000402, loss = 0.002032
grad_step = 000403, loss = 0.002036
grad_step = 000404, loss = 0.002032
grad_step = 000405, loss = 0.002020
grad_step = 000406, loss = 0.002013
grad_step = 000407, loss = 0.002017
grad_step = 000408, loss = 0.002021
grad_step = 000409, loss = 0.002021
grad_step = 000410, loss = 0.002019
grad_step = 000411, loss = 0.002017
grad_step = 000412, loss = 0.002011
grad_step = 000413, loss = 0.002008
grad_step = 000414, loss = 0.002008
grad_step = 000415, loss = 0.002011
grad_step = 000416, loss = 0.002011
grad_step = 000417, loss = 0.002009
grad_step = 000418, loss = 0.002008
grad_step = 000419, loss = 0.002006
grad_step = 000420, loss = 0.002003
grad_step = 000421, loss = 0.002001
grad_step = 000422, loss = 0.002001
grad_step = 000423, loss = 0.002000
grad_step = 000424, loss = 0.001999
grad_step = 000425, loss = 0.001997
grad_step = 000426, loss = 0.001997
grad_step = 000427, loss = 0.001996
grad_step = 000428, loss = 0.001995
grad_step = 000429, loss = 0.001994
grad_step = 000430, loss = 0.001993
grad_step = 000431, loss = 0.001993
grad_step = 000432, loss = 0.001992
grad_step = 000433, loss = 0.001991
grad_step = 000434, loss = 0.001990
grad_step = 000435, loss = 0.001989
grad_step = 000436, loss = 0.001988
grad_step = 000437, loss = 0.001987
grad_step = 000438, loss = 0.001986
grad_step = 000439, loss = 0.001985
grad_step = 000440, loss = 0.001984
grad_step = 000441, loss = 0.001983
grad_step = 000442, loss = 0.001983
grad_step = 000443, loss = 0.001982
grad_step = 000444, loss = 0.001981
grad_step = 000445, loss = 0.001981
grad_step = 000446, loss = 0.001980
grad_step = 000447, loss = 0.001981
grad_step = 000448, loss = 0.001982
grad_step = 000449, loss = 0.001985
grad_step = 000450, loss = 0.001993
grad_step = 000451, loss = 0.002010
grad_step = 000452, loss = 0.002045
grad_step = 000453, loss = 0.002116
grad_step = 000454, loss = 0.002258
grad_step = 000455, loss = 0.002516
grad_step = 000456, loss = 0.002944
grad_step = 000457, loss = 0.003381
grad_step = 000458, loss = 0.003524
grad_step = 000459, loss = 0.002911
grad_step = 000460, loss = 0.002132
grad_step = 000461, loss = 0.002053
grad_step = 000462, loss = 0.002567
grad_step = 000463, loss = 0.002824
grad_step = 000464, loss = 0.002387
grad_step = 000465, loss = 0.001987
grad_step = 000466, loss = 0.002250
grad_step = 000467, loss = 0.002530
grad_step = 000468, loss = 0.002290
grad_step = 000469, loss = 0.002004
grad_step = 000470, loss = 0.002151
grad_step = 000471, loss = 0.002341
grad_step = 000472, loss = 0.002151
grad_step = 000473, loss = 0.001990
grad_step = 000474, loss = 0.002116
grad_step = 000475, loss = 0.002170
grad_step = 000476, loss = 0.002046
grad_step = 000477, loss = 0.002001
grad_step = 000478, loss = 0.002070
grad_step = 000479, loss = 0.002067
grad_step = 000480, loss = 0.002005
grad_step = 000481, loss = 0.002009
grad_step = 000482, loss = 0.002032
grad_step = 000483, loss = 0.002011
grad_step = 000484, loss = 0.001992
grad_step = 000485, loss = 0.002000
grad_step = 000486, loss = 0.001994
grad_step = 000487, loss = 0.001983
grad_step = 000488, loss = 0.001990
grad_step = 000489, loss = 0.001980
grad_step = 000490, loss = 0.001965
grad_step = 000491, loss = 0.001974
grad_step = 000492, loss = 0.001975
grad_step = 000493, loss = 0.001959
grad_step = 000494, loss = 0.001953
grad_step = 000495, loss = 0.001964
grad_step = 000496, loss = 0.001963
grad_step = 000497, loss = 0.001945
grad_step = 000498, loss = 0.001948
grad_step = 000499, loss = 0.001959
grad_step = 000500, loss = 0.001948
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001937
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

  date_run                              2020-05-23 00:31:09.194365
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.239559
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-23 00:31:09.199901
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.130029
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-23 00:31:09.207097
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.152667
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-23 00:31:09.211851
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.975838
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
100%|██████████| 10/10 [00:02<00:00,  4.53it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 2.210 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.251176
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.251176071166992 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cf00e17f0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  9.08it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.102 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7c580a3f60> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:11<00:26,  3.80s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:22<00:14,  3.73s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:32<00:03,  3.64s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:35<00:00,  3.58s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 35.836 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.865953
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.865953493118286 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7c537fd588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.62it/s, avg_epoch_loss=5.81]
INFO:root:Epoch[0] Elapsed time 1.780 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.812772
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.812772130966186 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7c3c221e80> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [02:10<19:30, 130.00s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:04<19:06, 143.30s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [08:54<19:45, 169.42s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:24<18:09, 181.59s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [15:51<15:46, 189.23s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [19:49<13:35, 203.84s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [23:05<10:04, 201.44s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [27:30<07:20, 220.48s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [31:04<03:38, 218.70s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [34:52<00:00, 221.40s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [34:52<00:00, 209.27s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2092.728 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7c3c113160> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  6.07it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 1.672 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7c581a8cc0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 47.03it/s, avg_epoch_loss=5.09]
INFO:root:Epoch[0] Elapsed time 0.214 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.091111
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.091110706329346 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7c58231748> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json 

                        date_run  ...            metric_name
0   2020-05-23 00:30:41.644503  ...    mean_absolute_error
1   2020-05-23 00:30:41.648080  ...     mean_squared_error
2   2020-05-23 00:30:41.650833  ...  median_absolute_error
3   2020-05-23 00:30:41.653478  ...               r2_score
4   2020-05-23 00:30:52.988779  ...    mean_absolute_error
5   2020-05-23 00:30:52.992127  ...     mean_squared_error
6   2020-05-23 00:30:52.994673  ...  median_absolute_error
7   2020-05-23 00:30:52.997805  ...               r2_score
8   2020-05-23 00:31:09.194365  ...    mean_absolute_error
9   2020-05-23 00:31:09.199901  ...     mean_squared_error
10  2020-05-23 00:31:09.207097  ...  median_absolute_error
11  2020-05-23 00:31:09.211851  ...               r2_score

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
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do timeseries 





 ************************************************************************************************************************

  vision_mnist 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_cnn/mnist 

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0aa268d160> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fcba58> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fbc048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fcba58> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0aa268d160> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fcba58> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fbc048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fcba58> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0aa268d160> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a4d172d68> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0a49fbc048> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} 'data_info' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/cnn/mnist 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do vision_mnist 





 ************************************************************************************************************************

  fashion_vision_mnist 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 284, in <module>
    main()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 281, in main
    raise Exception("No options")
Exception: No options
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do fashion_vision_mnist 





 ************************************************************************************************************************

  text_classification 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text_classification/model_list_bench01.json 

  Model List [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}] 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f89e1781080> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b48026d90b3e75700c66426fe19a6e40a7104e78facce1995ddc533d0cddde9b
  Stored in directory: /tmp/pip-ephem-wheel-cache-keo5yn9b/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': True}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory. 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 153, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 291, in fit
    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 334, in get_dataset
    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 159, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f89814accc0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2637824/17464789 [===>..........................] - ETA: 0s
 8871936/17464789 [==============>...............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-23 01:08:30.795686: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 01:08:30.815472: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-23 01:08:30.815695: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cc112e2700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 01:08:30.815712: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5951 - accuracy: 0.5047
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5317 - accuracy: 0.5088
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5158 - accuracy: 0.5098
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5834 - accuracy: 0.5054
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5708 - accuracy: 0.5063
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5695 - accuracy: 0.5063
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5762 - accuracy: 0.5059
11000/25000 [============>.................] - ETA: 3s - loss: 7.5802 - accuracy: 0.5056
12000/25000 [=============>................] - ETA: 3s - loss: 7.5887 - accuracy: 0.5051
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6135 - accuracy: 0.5035
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6239 - accuracy: 0.5028
15000/25000 [=================>............] - ETA: 2s - loss: 7.6329 - accuracy: 0.5022
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6475 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6684 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6705 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 7s 272us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-23 01:08:43.897281
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-23 01:08:43.897281  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}] 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<142:12:13, 1.68kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<99:46:21, 2.40kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<69:52:59, 3.43kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<48:53:38, 4.89kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:05<34:07:17, 6.99kB/s].vector_cache/glove.6B.zip:   1%|          | 8.13M/862M [00:05<23:45:38, 9.98kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:05<16:33:38, 14.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.5M/862M [00:05<11:31:52, 20.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:05<8:01:57, 29.1kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:05<5:35:38, 41.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:06<3:53:51, 59.3kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:06<2:42:54, 84.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:06<1:53:34, 121kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:06<1:19:07, 173kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:06<55:14, 246kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:06<38:33, 351kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:07<28:47, 469kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:09<21:58, 611kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:09<17:35, 763kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:09<12:47, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:11<11:16, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:11<09:34, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:11<07:04, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:13<07:34, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:13<08:27, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:13<06:41, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:13<04:51, 2.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:15<14:30, 911kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:15<11:30, 1.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:15<08:23, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:17<08:55, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:17<07:37, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:17<05:39, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:18<07:00, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:19<06:01, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:19<04:28, 2.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:19<03:18, 3.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:20<51:27, 253kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:21<37:20, 348kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:21<26:24, 491kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:22<21:29, 602kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:23<16:22, 790kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:23<11:46, 1.10MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:24<11:15, 1.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:25<09:12, 1.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:25<06:45, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:26<07:44, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:26<08:03, 1.59MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:27<06:10, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:27<04:31, 2.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:28<08:06, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:28<06:59, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:29<05:12, 2.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<06:35, 1.92MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:56, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:31<04:29, 2.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<06:05, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<05:34, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:32<04:13, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<05:11, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<03:54, 3.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<02:53, 4.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:37<1:03:42, 196kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:37<46:19, 269kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:38<33:39, 368kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<24:56, 497kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<17:46, 696kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<15:04, 818kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<13:11, 934kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<09:46, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<06:59, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<10:46, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<08:49, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<06:26, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<07:21, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<06:24, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:44<04:47, 2.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<06:12, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:35, 2.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<04:09, 2.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<05:46, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<05:17, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<04:00, 3.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<05:37, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<05:10, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<03:55, 3.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<05:33, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<05:06, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<03:49, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<05:04, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<03:50, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<05:27, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<03:49, 3.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<04:59, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<03:47, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<04:58, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<03:46, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<05:22, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<04:56, 2.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<03:44, 3.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<05:20, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:04<04:55, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<03:43, 3.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:06<04:53, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<03:40, 3.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<05:16, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<03:41, 3.08MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<05:59, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:10<04:42, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<03:24, 3.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<12:06, 927kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<09:39, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<06:59, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<07:29, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<06:23, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<04:45, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<05:56, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<03:55, 2.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<05:22, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<04:53, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:18<03:38, 3.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<05:09, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<04:43, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<03:32, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<05:04, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<04:40, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:22<03:32, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<05:01, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<05:45, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:24<04:29, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<03:17, 3.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<06:50, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<05:42, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<04:13, 2.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<03:05, 3.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<45:08, 237kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<33:47, 316kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<24:04, 443kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:28<16:57, 627kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<16:18, 651kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<12:31, 847kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<09:00, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<08:45, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<08:18, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<06:16, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<04:29, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<11:36, 902kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<09:11, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<06:41, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<07:06, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<06:02, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<04:27, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:32, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<06:00, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<04:43, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<03:24, 3.02MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<9:54:09, 17.3kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<6:56:44, 24.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<4:51:13, 35.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<3:25:31, 49.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<2:24:51, 70.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<1:41:23, 100kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:43<1:13:06, 139kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<53:14, 190kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<37:44, 268kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<27:55, 360kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<20:34, 489kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<14:35, 688kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<12:31, 798kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<09:46, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<07:05, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<07:17, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<06:06, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<05:29, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<05:52, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<04:31, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<03:21, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<05:09, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<04:36, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<03:28, 2.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<04:41, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<05:23, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<04:16, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<03:06, 3.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<1:10:51, 136kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<50:34, 191kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<35:31, 271kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<27:00, 355kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<19:52, 482kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<14:07, 676kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<12:05, 786kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<09:26, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<06:50, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<07:00, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:51, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:03<04:19, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<05:14, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:05<03:28, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<04:37, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<05:08, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<04:04, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<04:20, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<03:59, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<03:01, 3.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<04:14, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<04:56, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<03:53, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<02:48, 3.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<12:53, 706kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<09:58, 911kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<07:10, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<07:06, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<05:54, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<04:21, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<05:09, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<05:27, 1.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<04:12, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<03:04, 2.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<05:32, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<04:47, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<03:34, 2.48MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<04:33, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<04:04, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:20<03:04, 2.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<04:12, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<03:51, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<02:54, 3.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<04:05, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<03:45, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<02:48, 3.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<04:00, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<04:34, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<03:38, 2.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<03:55, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<03:38, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<02:45, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<03:35, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<02:43, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<02:24, 3.51MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<6:47:01, 20.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<4:44:53, 29.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<3:18:28, 42.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<2:25:05, 57.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<1:43:24, 80.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<1:12:40, 115kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<50:48, 163kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<38:25, 215kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<27:46, 298kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<19:36, 420kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<15:29, 529kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<12:37, 650kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<09:16, 883kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<06:33, 1.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<17:20, 469kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<13:01, 624kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<09:18, 870kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<08:17, 973kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<07:31, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<05:39, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:42<04:02, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<07:09, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<05:52, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<04:19, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<04:46, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<05:02, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<03:54, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<02:48, 2.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<08:05, 972kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<06:31, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:48<04:45, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<05:03, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<05:13, 1.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<04:01, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<02:54, 2.67MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<05:59, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<05:02, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:52<03:43, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:17, 1.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:43, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:39, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<02:39, 2.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<05:13, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<04:19, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<03:11, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<02:20, 3.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<10:01, 749kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<08:42, 863kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<06:26, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<04:35, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<06:59, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<05:42, 1.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<04:10, 1.77MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<04:32, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<04:49, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<03:42, 1.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<02:45, 2.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<03:47, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<03:26, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<02:34, 2.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<03:24, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:09, 2.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<02:22, 3.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<03:15, 2.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<03:53, 1.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<02:15, 3.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:48, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:25, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<02:35, 2.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<03:22, 2.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<03:55, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:12<03:04, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<02:14, 3.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<04:35, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<03:58, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<02:55, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<03:34, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<04:04, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<03:09, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<02:20, 2.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:18<03:32, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<03:12, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<02:25, 2.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<03:11, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<03:41, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<02:53, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<02:07, 3.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<04:01, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<03:32, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<02:39, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<03:19, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<03:45, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<02:58, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:24<02:09, 3.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<21:06, 310kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<15:27, 422kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<10:57, 594kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:28<09:04, 713kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:28<07:02, 918kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<05:05, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<04:57, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<04:54, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<03:44, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<02:41, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:32<05:21, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<04:25, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<03:15, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:34<03:36, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:34<03:56, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<03:05, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<02:14, 2.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<11:14, 550kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<08:31, 725kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<06:06, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<05:36, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<05:15, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:38<03:56, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<02:50, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:40<04:30, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:49, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<02:49, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:18, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:39, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:42<02:52, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:04, 2.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<10:42, 551kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<08:07, 726kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<05:49, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<05:21, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<05:03, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:46<03:50, 1.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<02:44, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<10:35, 545kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<08:01, 718kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<05:44, 998kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:50<05:16, 1.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:50<04:54, 1.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<03:41, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<02:39, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:52<03:56, 1.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<03:21, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<02:29, 2.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<02:59, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<03:14, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<02:30, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<01:48, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<05:15, 1.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<04:16, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<03:07, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<03:23, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<03:32, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<02:43, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<01:58, 2.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<04:04, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<03:25, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:31, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<02:56, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<03:12, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<01:47, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<05:23, 968kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<04:19, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<03:09, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<03:20, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<03:26, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<02:40, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<01:54, 2.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:07<30:33, 166kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<21:55, 231kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<15:24, 327kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<11:49, 424kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<08:48, 568kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<06:14, 796kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<05:26, 906kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<04:49, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<03:35, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:33, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:13<05:46, 843kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<04:34, 1.06MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<03:18, 1.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<03:22, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<02:52, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<02:07, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<02:46, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<02:10, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:18<01:34, 2.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<06:47, 686kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<05:14, 889kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<03:45, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<03:40, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<03:04, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:16, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<02:35, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<02:46, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<02:09, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<01:34, 2.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<02:32, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<02:15, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<01:41, 2.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:27<02:09, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:27<02:26, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<01:54, 2.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<01:23, 3.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<02:38, 1.63MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<02:18, 1.87MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<01:42, 2.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<02:10, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:31<01:57, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<01:27, 2.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<01:59, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<01:50, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<01:22, 3.00MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<01:52, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<01:44, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<01:19, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<01:51, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<01:44, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<01:19, 3.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<01:48, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<02:06, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:39<01:39, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<01:12, 3.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<02:56, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<02:29, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<01:49, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<02:06, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<02:09, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<01:42, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<01:14, 3.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<02:38, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<02:14, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<01:38, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<01:59, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<01:46, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:47<01:19, 2.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<01:43, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<01:59, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<01:35, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<01:08, 3.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<06:08, 578kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<04:40, 759kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<03:20, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<02:31, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<2:39:49, 21.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<1:51:39, 31.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<1:17:02, 44.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<1:00:09, 56.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<42:48, 79.7kB/s]  .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<30:01, 113kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<20:44, 161kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<38:16, 87.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<27:06, 123kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<18:54, 175kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:59<13:50, 237kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<10:22, 316kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<07:23, 441kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<05:10, 624kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<04:43, 680kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<03:38, 878kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<02:37, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<02:30, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<02:24, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<01:50, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<01:18, 2.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<16:58, 181kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<12:11, 251kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<08:32, 356kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<06:34, 456kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<04:54, 610kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<03:29, 852kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<03:05, 949kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<02:49, 1.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<02:06, 1.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:09<01:29, 1.93MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<03:16, 876kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<02:35, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:52, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:55, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:38, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:12, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:26, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:37, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:16, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<00:54, 2.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<03:57, 671kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<03:02, 869kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<02:11, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<02:05, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<02:02, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:32, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:06, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:21<01:42, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:27, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:04, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:17, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<01:25, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<01:07, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<00:48, 2.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<17:29, 136kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<12:28, 190kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<08:41, 270kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<06:30, 354kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<05:04, 455kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<03:39, 627kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<02:31, 886kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<05:08, 436kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<03:49, 584kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<02:41, 818kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<02:20, 926kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<02:06, 1.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:33, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<01:06, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:40, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:23, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:01, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<01:09, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<01:14, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<00:57, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<00:40, 2.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<01:52, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<01:31, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<01:06, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<01:10, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<01:13, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:56, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<00:40, 2.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:41<01:20, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:41<01:07, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:49, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<00:58, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:43<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<00:38, 2.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<00:48, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:45<00:55, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<00:44, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<00:31, 3.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:46<02:22, 678kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<01:49, 879kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:18, 1.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<01:14, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<01:12, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:54, 1.70MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:38, 2.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<00:58, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<00:50, 1.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<00:37, 2.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:44, 1.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:28, 2.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<00:20, 3.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<01:45, 762kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:55<01:30, 882kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<01:07, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:55<00:46, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<02:32, 501kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<01:54, 663kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<01:20, 924kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<01:10, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:56, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:40, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<00:42, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:45, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:01<00:34, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:23, 2.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<01:02, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:50, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<00:35, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:38, 1.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:39, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:30, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:05<00:20, 2.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<01:39, 558kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<01:14, 733kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:52, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:46, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:43, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:32, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:22, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:37, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:30, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:21, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<00:24, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:21, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:15, 2.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:14<00:18, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:21, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:16, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:11, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:20, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:17, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:12, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:15, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:16, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:13, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<00:08, 3.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<02:12, 199kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<01:34, 275kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<01:02, 389kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:44, 494kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:35, 612kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:25, 840kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:16, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:16, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:13, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:09, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:08, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:08, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:26<00:06, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:03, 2.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:08, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:06, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:04, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:03, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:03, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:02, 2.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:00, 2.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:02, 562kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:01, 740kB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.19MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<207:00:21,  1.86s/it]  0%|          | 1052/400000 [00:01<144:31:35,  1.30s/it]  1%|          | 2113/400000 [00:02<100:54:09,  1.10it/s]  1%|          | 3200/400000 [00:02<70:26:31,  1.56it/s]   1%|          | 4342/400000 [00:02<49:10:13,  2.24it/s]  1%|▏         | 5449/400000 [00:02<34:19:33,  3.19it/s]  2%|▏         | 6607/400000 [00:02<23:57:37,  4.56it/s]  2%|▏         | 7665/400000 [00:02<16:43:49,  6.51it/s]  2%|▏         | 8727/400000 [00:02<11:40:57,  9.30it/s]  2%|▏         | 9849/400000 [00:02<8:09:26, 13.29it/s]   3%|▎         | 10903/400000 [00:02<5:41:52, 18.97it/s]  3%|▎         | 11998/400000 [00:02<3:58:48, 27.08it/s]  3%|▎         | 13055/400000 [00:03<2:46:53, 38.64it/s]  4%|▎         | 14112/400000 [00:03<1:56:42, 55.11it/s]  4%|▍         | 15136/400000 [00:03<1:21:40, 78.53it/s]  4%|▍         | 16147/400000 [00:03<57:13, 111.81it/s]   4%|▍         | 17168/400000 [00:03<40:08, 158.98it/s]  5%|▍         | 18178/400000 [00:03<28:13, 225.48it/s]  5%|▍         | 19167/400000 [00:03<19:54, 318.93it/s]  5%|▌         | 20150/400000 [00:03<14:05, 449.24it/s]  5%|▌         | 21183/400000 [00:03<10:01, 630.03it/s]  6%|▌         | 22177/400000 [00:03<07:11, 876.22it/s]  6%|▌         | 23171/400000 [00:04<05:13, 1202.27it/s]  6%|▌         | 24141/400000 [00:04<03:50, 1627.12it/s]  6%|▋         | 25133/400000 [00:04<02:52, 2171.68it/s]  7%|▋         | 26172/400000 [00:04<02:11, 2847.27it/s]  7%|▋         | 27267/400000 [00:04<01:41, 3659.56it/s]  7%|▋         | 28288/400000 [00:04<01:22, 4493.56it/s]  7%|▋         | 29332/400000 [00:04<01:08, 5419.43it/s]  8%|▊         | 30398/400000 [00:04<00:58, 6356.23it/s]  8%|▊         | 31488/400000 [00:04<00:50, 7264.47it/s]  8%|▊         | 32564/400000 [00:05<00:45, 8047.72it/s]  8%|▊         | 33621/400000 [00:05<00:43, 8470.19it/s]  9%|▊         | 34689/400000 [00:05<00:40, 9030.15it/s]  9%|▉         | 35767/400000 [00:05<00:38, 9492.09it/s]  9%|▉         | 36819/400000 [00:05<00:38, 9470.33it/s]  9%|▉         | 37915/400000 [00:05<00:36, 9870.83it/s] 10%|▉         | 38957/400000 [00:05<00:36, 10000.92it/s] 10%|▉         | 39996/400000 [00:05<00:36, 9980.74it/s]  10%|█         | 41022/400000 [00:05<00:35, 10048.06it/s] 11%|█         | 42046/400000 [00:05<00:35, 10049.15it/s] 11%|█         | 43091/400000 [00:06<00:35, 10165.17it/s] 11%|█         | 44118/400000 [00:06<00:35, 9946.94it/s]  11%|█▏        | 45138/400000 [00:06<00:35, 10020.39it/s] 12%|█▏        | 46147/400000 [00:06<00:35, 10040.99it/s] 12%|█▏        | 47219/400000 [00:06<00:34, 10234.64it/s] 12%|█▏        | 48247/400000 [00:06<00:34, 10182.34it/s] 12%|█▏        | 49268/400000 [00:06<00:34, 10172.44it/s] 13%|█▎        | 50293/400000 [00:06<00:34, 10193.74it/s] 13%|█▎        | 51345/400000 [00:06<00:33, 10288.62it/s] 13%|█▎        | 52375/400000 [00:06<00:34, 10195.42it/s] 13%|█▎        | 53464/400000 [00:07<00:33, 10393.96it/s] 14%|█▎        | 54506/400000 [00:07<00:33, 10308.63it/s] 14%|█▍        | 55587/400000 [00:07<00:32, 10452.94it/s] 14%|█▍        | 56634/400000 [00:07<00:33, 10256.84it/s] 14%|█▍        | 57662/400000 [00:07<00:34, 9803.76it/s]  15%|█▍        | 58648/400000 [00:07<00:35, 9530.45it/s] 15%|█▍        | 59607/400000 [00:07<00:36, 9390.56it/s] 15%|█▌        | 60573/400000 [00:07<00:35, 9468.06it/s] 15%|█▌        | 61523/400000 [00:07<00:36, 9327.34it/s] 16%|█▌        | 62485/400000 [00:08<00:35, 9411.63it/s] 16%|█▌        | 63506/400000 [00:08<00:34, 9635.13it/s] 16%|█▌        | 64473/400000 [00:08<00:35, 9435.06it/s] 16%|█▋        | 65443/400000 [00:08<00:35, 9511.02it/s] 17%|█▋        | 66404/400000 [00:08<00:34, 9539.93it/s] 17%|█▋        | 67360/400000 [00:08<00:35, 9471.52it/s] 17%|█▋        | 68309/400000 [00:08<00:35, 9401.44it/s] 17%|█▋        | 69254/400000 [00:08<00:35, 9414.47it/s] 18%|█▊        | 70197/400000 [00:08<00:36, 9139.20it/s] 18%|█▊        | 71257/400000 [00:08<00:34, 9532.93it/s] 18%|█▊        | 72300/400000 [00:09<00:33, 9785.03it/s] 18%|█▊        | 73361/400000 [00:09<00:32, 10017.91it/s] 19%|█▊        | 74420/400000 [00:09<00:31, 10181.32it/s] 19%|█▉        | 75497/400000 [00:09<00:31, 10350.51it/s] 19%|█▉        | 76594/400000 [00:09<00:30, 10526.82it/s] 19%|█▉        | 77651/400000 [00:09<00:30, 10454.10it/s] 20%|█▉        | 78699/400000 [00:09<00:30, 10419.32it/s] 20%|█▉        | 79743/400000 [00:09<00:31, 10317.37it/s] 20%|██        | 80777/400000 [00:09<00:31, 10057.74it/s] 20%|██        | 81786/400000 [00:09<00:32, 9853.79it/s]  21%|██        | 82839/400000 [00:10<00:31, 10045.26it/s] 21%|██        | 83847/400000 [00:10<00:31, 10009.44it/s] 21%|██        | 84850/400000 [00:10<00:31, 9867.69it/s]  21%|██▏       | 85839/400000 [00:10<00:32, 9681.59it/s] 22%|██▏       | 86810/400000 [00:10<00:32, 9543.87it/s] 22%|██▏       | 87767/400000 [00:10<00:32, 9529.41it/s] 22%|██▏       | 88823/400000 [00:10<00:31, 9816.22it/s] 22%|██▏       | 89808/400000 [00:10<00:31, 9708.60it/s] 23%|██▎       | 90782/400000 [00:10<00:32, 9658.54it/s] 23%|██▎       | 91750/400000 [00:10<00:32, 9627.19it/s] 23%|██▎       | 92714/400000 [00:11<00:32, 9473.67it/s] 23%|██▎       | 93675/400000 [00:11<00:32, 9514.01it/s] 24%|██▎       | 94702/400000 [00:11<00:31, 9727.43it/s] 24%|██▍       | 95688/400000 [00:11<00:31, 9765.50it/s] 24%|██▍       | 96709/400000 [00:11<00:30, 9894.30it/s] 24%|██▍       | 97762/400000 [00:11<00:30, 10074.60it/s] 25%|██▍       | 98790/400000 [00:11<00:29, 10131.89it/s] 25%|██▍       | 99805/400000 [00:11<00:29, 10080.48it/s] 25%|██▌       | 100854/400000 [00:11<00:29, 10198.81it/s] 25%|██▌       | 101917/400000 [00:11<00:28, 10322.43it/s] 26%|██▌       | 103000/400000 [00:12<00:28, 10468.46it/s] 26%|██▌       | 104101/400000 [00:12<00:27, 10624.84it/s] 26%|██▋       | 105165/400000 [00:12<00:28, 10526.62it/s] 27%|██▋       | 106283/400000 [00:12<00:27, 10712.79it/s] 27%|██▋       | 107397/400000 [00:12<00:26, 10837.15it/s] 27%|██▋       | 108483/400000 [00:12<00:27, 10718.24it/s] 27%|██▋       | 109557/400000 [00:12<00:28, 10342.69it/s] 28%|██▊       | 110596/400000 [00:12<00:28, 10167.56it/s] 28%|██▊       | 111617/400000 [00:12<00:28, 10086.69it/s] 28%|██▊       | 112693/400000 [00:13<00:27, 10277.81it/s] 28%|██▊       | 113749/400000 [00:13<00:27, 10360.30it/s] 29%|██▊       | 114788/400000 [00:13<00:28, 10052.28it/s] 29%|██▉       | 115814/400000 [00:13<00:28, 10113.39it/s] 29%|██▉       | 116880/400000 [00:13<00:27, 10270.84it/s] 29%|██▉       | 117910/400000 [00:13<00:28, 9968.55it/s]  30%|██▉       | 119021/400000 [00:13<00:27, 10283.22it/s] 30%|███       | 120055/400000 [00:13<00:27, 10212.07it/s] 30%|███       | 121080/400000 [00:13<00:27, 10213.03it/s] 31%|███       | 122106/400000 [00:13<00:27, 10226.69it/s] 31%|███       | 123182/400000 [00:14<00:26, 10378.57it/s] 31%|███       | 124236/400000 [00:14<00:26, 10424.59it/s] 31%|███▏      | 125305/400000 [00:14<00:26, 10499.90it/s] 32%|███▏      | 126357/400000 [00:14<00:26, 10343.86it/s] 32%|███▏      | 127423/400000 [00:14<00:26, 10433.64it/s] 32%|███▏      | 128468/400000 [00:14<00:26, 10092.59it/s] 32%|███▏      | 129481/400000 [00:14<00:27, 9867.81it/s]  33%|███▎      | 130472/400000 [00:14<00:27, 9675.30it/s] 33%|███▎      | 131455/400000 [00:14<00:27, 9720.57it/s] 33%|███▎      | 132492/400000 [00:14<00:27, 9904.20it/s] 33%|███▎      | 133494/400000 [00:15<00:26, 9935.25it/s] 34%|███▎      | 134521/400000 [00:15<00:26, 10032.32it/s] 34%|███▍      | 135579/400000 [00:15<00:25, 10189.63it/s] 34%|███▍      | 136600/400000 [00:15<00:25, 10140.58it/s] 34%|███▍      | 137651/400000 [00:15<00:25, 10248.24it/s] 35%|███▍      | 138677/400000 [00:15<00:25, 10238.87it/s] 35%|███▍      | 139702/400000 [00:15<00:25, 10144.76it/s] 35%|███▌      | 140757/400000 [00:15<00:25, 10261.67it/s] 35%|███▌      | 141785/400000 [00:15<00:25, 10060.73it/s] 36%|███▌      | 142810/400000 [00:15<00:25, 10113.44it/s] 36%|███▌      | 143825/400000 [00:16<00:25, 10122.71it/s] 36%|███▌      | 144839/400000 [00:16<00:25, 10037.41it/s] 36%|███▋      | 145904/400000 [00:16<00:24, 10211.36it/s] 37%|███▋      | 146927/400000 [00:16<00:24, 10192.29it/s] 37%|███▋      | 147948/400000 [00:16<00:24, 10139.37it/s] 37%|███▋      | 148963/400000 [00:16<00:24, 10073.83it/s] 37%|███▋      | 149971/400000 [00:16<00:24, 10066.52it/s] 38%|███▊      | 150979/400000 [00:16<00:24, 10069.02it/s] 38%|███▊      | 151987/400000 [00:16<00:25, 9641.72it/s]  38%|███▊      | 152984/400000 [00:17<00:25, 9735.23it/s] 38%|███▊      | 154000/400000 [00:17<00:24, 9856.47it/s] 39%|███▉      | 155013/400000 [00:17<00:24, 9935.38it/s] 39%|███▉      | 156056/400000 [00:17<00:24, 10076.48it/s] 39%|███▉      | 157066/400000 [00:17<00:24, 10043.59it/s] 40%|███▉      | 158072/400000 [00:17<00:24, 9950.08it/s]  40%|███▉      | 159093/400000 [00:17<00:24, 10025.51it/s] 40%|████      | 160166/400000 [00:17<00:23, 10226.55it/s] 40%|████      | 161191/400000 [00:17<00:23, 10177.92it/s] 41%|████      | 162210/400000 [00:17<00:23, 10090.18it/s] 41%|████      | 163221/400000 [00:18<00:23, 9934.12it/s]  41%|████      | 164216/400000 [00:18<00:24, 9728.74it/s] 41%|████▏     | 165273/400000 [00:18<00:23, 9966.57it/s] 42%|████▏     | 166318/400000 [00:18<00:23, 10105.53it/s] 42%|████▏     | 167394/400000 [00:18<00:22, 10291.18it/s] 42%|████▏     | 168512/400000 [00:18<00:21, 10541.13it/s] 42%|████▏     | 169636/400000 [00:18<00:21, 10739.95it/s] 43%|████▎     | 170763/400000 [00:18<00:21, 10892.61it/s] 43%|████▎     | 171878/400000 [00:18<00:20, 10966.61it/s] 43%|████▎     | 172977/400000 [00:18<00:21, 10736.86it/s] 44%|████▎     | 174054/400000 [00:19<00:21, 10674.02it/s] 44%|████▍     | 175179/400000 [00:19<00:20, 10839.96it/s] 44%|████▍     | 176265/400000 [00:19<00:20, 10745.94it/s] 44%|████▍     | 177385/400000 [00:19<00:20, 10877.76it/s] 45%|████▍     | 178475/400000 [00:19<00:20, 10685.45it/s] 45%|████▍     | 179546/400000 [00:19<00:20, 10590.85it/s] 45%|████▌     | 180623/400000 [00:19<00:20, 10643.38it/s] 45%|████▌     | 181689/400000 [00:19<00:21, 10353.16it/s] 46%|████▌     | 182727/400000 [00:19<00:21, 10244.10it/s] 46%|████▌     | 183754/400000 [00:19<00:21, 9912.95it/s]  46%|████▌     | 184750/400000 [00:20<00:22, 9747.25it/s] 46%|████▋     | 185728/400000 [00:20<00:22, 9643.77it/s] 47%|████▋     | 186847/400000 [00:20<00:21, 10059.22it/s] 47%|████▋     | 187867/400000 [00:20<00:21, 10097.24it/s] 47%|████▋     | 188890/400000 [00:20<00:20, 10136.01it/s] 47%|████▋     | 189907/400000 [00:20<00:20, 10131.73it/s] 48%|████▊     | 190930/400000 [00:20<00:20, 10160.01it/s] 48%|████▊     | 191976/400000 [00:20<00:20, 10247.74it/s] 48%|████▊     | 193003/400000 [00:20<00:20, 10189.80it/s] 49%|████▊     | 194073/400000 [00:20<00:19, 10336.21it/s] 49%|████▉     | 195197/400000 [00:21<00:19, 10589.35it/s] 49%|████▉     | 196259/400000 [00:21<00:19, 10305.89it/s] 49%|████▉     | 197293/400000 [00:21<00:19, 10146.97it/s] 50%|████▉     | 198318/400000 [00:21<00:19, 10175.35it/s] 50%|████▉     | 199418/400000 [00:21<00:19, 10408.87it/s] 50%|█████     | 200571/400000 [00:21<00:18, 10719.63it/s] 50%|█████     | 201659/400000 [00:21<00:18, 10765.50it/s] 51%|█████     | 202746/400000 [00:21<00:18, 10794.82it/s] 51%|█████     | 203828/400000 [00:21<00:18, 10777.06it/s] 51%|█████     | 204908/400000 [00:22<00:18, 10501.17it/s] 52%|█████▏    | 206026/400000 [00:22<00:18, 10694.14it/s] 52%|█████▏    | 207129/400000 [00:22<00:17, 10791.17it/s] 52%|█████▏    | 208211/400000 [00:22<00:18, 10529.11it/s] 52%|█████▏    | 209267/400000 [00:22<00:18, 10401.62it/s] 53%|█████▎    | 210310/400000 [00:22<00:18, 10163.04it/s] 53%|█████▎    | 211418/400000 [00:22<00:18, 10419.71it/s] 53%|█████▎    | 212464/400000 [00:22<00:18, 10225.40it/s] 53%|█████▎    | 213490/400000 [00:22<00:18, 10138.64it/s] 54%|█████▎    | 214507/400000 [00:22<00:18, 9963.72it/s]  54%|█████▍    | 215506/400000 [00:23<00:19, 9637.01it/s] 54%|█████▍    | 216474/400000 [00:23<00:19, 9291.46it/s] 54%|█████▍    | 217409/400000 [00:23<00:19, 9172.17it/s] 55%|█████▍    | 218331/400000 [00:23<00:20, 8962.90it/s] 55%|█████▍    | 219239/400000 [00:23<00:20, 8995.39it/s] 55%|█████▌    | 220142/400000 [00:23<00:19, 8998.46it/s] 55%|█████▌    | 221061/400000 [00:23<00:19, 9053.52it/s] 55%|█████▌    | 221968/400000 [00:23<00:20, 8862.37it/s] 56%|█████▌    | 222857/400000 [00:23<00:20, 8788.03it/s] 56%|█████▌    | 223738/400000 [00:24<00:21, 8355.81it/s] 56%|█████▌    | 224635/400000 [00:24<00:20, 8529.38it/s] 56%|█████▋    | 225546/400000 [00:24<00:20, 8695.53it/s] 57%|█████▋    | 226495/400000 [00:24<00:19, 8919.24it/s] 57%|█████▋    | 227414/400000 [00:24<00:19, 8996.92it/s] 57%|█████▋    | 228323/400000 [00:24<00:19, 9023.90it/s] 57%|█████▋    | 229244/400000 [00:24<00:18, 9076.14it/s] 58%|█████▊    | 230154/400000 [00:24<00:18, 8956.81it/s] 58%|█████▊    | 231077/400000 [00:24<00:18, 9035.60it/s] 58%|█████▊    | 231982/400000 [00:24<00:18, 8844.14it/s] 58%|█████▊    | 232869/400000 [00:25<00:19, 8580.05it/s] 58%|█████▊    | 233762/400000 [00:25<00:19, 8679.86it/s] 59%|█████▊    | 234634/400000 [00:25<00:19, 8689.55it/s] 59%|█████▉    | 235581/400000 [00:25<00:18, 8909.61it/s] 59%|█████▉    | 236475/400000 [00:25<00:18, 8891.28it/s] 59%|█████▉    | 237366/400000 [00:25<00:18, 8858.77it/s] 60%|█████▉    | 238254/400000 [00:25<00:18, 8828.92it/s] 60%|█████▉    | 239138/400000 [00:25<00:18, 8811.39it/s] 60%|██████    | 240046/400000 [00:25<00:17, 8890.08it/s] 60%|██████    | 240936/400000 [00:25<00:18, 8822.07it/s] 60%|██████    | 241819/400000 [00:26<00:18, 8353.33it/s] 61%|██████    | 242711/400000 [00:26<00:18, 8515.11it/s] 61%|██████    | 243568/400000 [00:26<00:18, 8493.48it/s] 61%|██████    | 244421/400000 [00:26<00:19, 8131.10it/s] 61%|██████▏   | 245264/400000 [00:26<00:18, 8217.88it/s] 62%|██████▏   | 246118/400000 [00:26<00:18, 8305.76it/s] 62%|██████▏   | 246952/400000 [00:26<00:19, 7945.25it/s] 62%|██████▏   | 247802/400000 [00:26<00:18, 8102.59it/s] 62%|██████▏   | 248726/400000 [00:26<00:17, 8411.17it/s] 62%|██████▏   | 249615/400000 [00:27<00:17, 8547.36it/s] 63%|██████▎   | 250517/400000 [00:27<00:17, 8681.41it/s] 63%|██████▎   | 251390/400000 [00:27<00:17, 8688.34it/s] 63%|██████▎   | 252276/400000 [00:27<00:16, 8737.24it/s] 63%|██████▎   | 253254/400000 [00:27<00:16, 9025.93it/s] 64%|██████▎   | 254191/400000 [00:27<00:15, 9125.24it/s] 64%|██████▍   | 255148/400000 [00:27<00:15, 9252.04it/s] 64%|██████▍   | 256076/400000 [00:27<00:15, 9014.01it/s] 64%|██████▍   | 256981/400000 [00:27<00:16, 8690.06it/s] 64%|██████▍   | 257855/400000 [00:27<00:16, 8580.33it/s] 65%|██████▍   | 258717/400000 [00:28<00:16, 8369.01it/s] 65%|██████▍   | 259597/400000 [00:28<00:16, 8488.03it/s] 65%|██████▌   | 260518/400000 [00:28<00:16, 8691.49it/s] 65%|██████▌   | 261428/400000 [00:28<00:15, 8808.06it/s] 66%|██████▌   | 262388/400000 [00:28<00:15, 9029.51it/s] 66%|██████▌   | 263321/400000 [00:28<00:14, 9114.97it/s] 66%|██████▌   | 264235/400000 [00:28<00:14, 9054.82it/s] 66%|██████▋   | 265143/400000 [00:28<00:14, 9034.64it/s] 67%|██████▋   | 266078/400000 [00:28<00:14, 9126.00it/s] 67%|██████▋   | 267060/400000 [00:28<00:14, 9322.78it/s] 67%|██████▋   | 268027/400000 [00:29<00:14, 9422.15it/s] 67%|██████▋   | 268971/400000 [00:29<00:14, 9256.00it/s] 67%|██████▋   | 269899/400000 [00:29<00:14, 9028.92it/s] 68%|██████▊   | 270805/400000 [00:29<00:14, 9020.50it/s] 68%|██████▊   | 271709/400000 [00:29<00:14, 8997.51it/s] 68%|██████▊   | 272610/400000 [00:29<00:14, 8906.13it/s] 68%|██████▊   | 273510/400000 [00:29<00:14, 8933.07it/s] 69%|██████▊   | 274493/400000 [00:29<00:13, 9183.90it/s] 69%|██████▉   | 275470/400000 [00:29<00:13, 9351.09it/s] 69%|██████▉   | 276450/400000 [00:29<00:13, 9480.41it/s] 69%|██████▉   | 277401/400000 [00:30<00:13, 9429.36it/s] 70%|██████▉   | 278368/400000 [00:30<00:12, 9498.65it/s] 70%|██████▉   | 279373/400000 [00:30<00:12, 9655.33it/s] 70%|███████   | 280431/400000 [00:30<00:12, 9914.72it/s] 70%|███████   | 281485/400000 [00:30<00:11, 10093.21it/s] 71%|███████   | 282498/400000 [00:30<00:11, 9951.28it/s]  71%|███████   | 283504/400000 [00:30<00:11, 9982.86it/s] 71%|███████   | 284525/400000 [00:30<00:11, 10049.26it/s] 71%|███████▏  | 285561/400000 [00:30<00:11, 10139.52it/s] 72%|███████▏  | 286577/400000 [00:30<00:11, 10128.94it/s] 72%|███████▏  | 287591/400000 [00:31<00:11, 9939.47it/s]  72%|███████▏  | 288587/400000 [00:31<00:11, 9768.46it/s] 72%|███████▏  | 289566/400000 [00:31<00:11, 9488.75it/s] 73%|███████▎  | 290518/400000 [00:31<00:11, 9388.99it/s] 73%|███████▎  | 291568/400000 [00:31<00:11, 9696.79it/s] 73%|███████▎  | 292542/400000 [00:31<00:11, 9292.20it/s] 73%|███████▎  | 293478/400000 [00:31<00:11, 9197.77it/s] 74%|███████▎  | 294450/400000 [00:31<00:11, 9347.09it/s] 74%|███████▍  | 295494/400000 [00:31<00:10, 9650.10it/s] 74%|███████▍  | 296559/400000 [00:32<00:10, 9927.60it/s] 74%|███████▍  | 297558/400000 [00:32<00:10, 9569.26it/s] 75%|███████▍  | 298588/400000 [00:32<00:10, 9776.65it/s] 75%|███████▍  | 299625/400000 [00:32<00:10, 9946.30it/s] 75%|███████▌  | 300709/400000 [00:32<00:09, 10198.10it/s] 75%|███████▌  | 301758/400000 [00:32<00:09, 10283.25it/s] 76%|███████▌  | 302790/400000 [00:32<00:09, 10184.63it/s] 76%|███████▌  | 303817/400000 [00:32<00:09, 10206.73it/s] 76%|███████▌  | 304840/400000 [00:32<00:09, 9706.62it/s]  76%|███████▋  | 305818/400000 [00:32<00:10, 9147.40it/s] 77%|███████▋  | 306744/400000 [00:33<00:10, 9168.25it/s] 77%|███████▋  | 307683/400000 [00:33<00:10, 9231.08it/s] 77%|███████▋  | 308648/400000 [00:33<00:09, 9352.83it/s] 77%|███████▋  | 309653/400000 [00:33<00:09, 9549.81it/s] 78%|███████▊  | 310661/400000 [00:33<00:09, 9700.64it/s] 78%|███████▊  | 311646/400000 [00:33<00:09, 9744.83it/s] 78%|███████▊  | 312623/400000 [00:33<00:09, 9628.38it/s] 78%|███████▊  | 313604/400000 [00:33<00:08, 9680.22it/s] 79%|███████▊  | 314595/400000 [00:33<00:08, 9745.43it/s] 79%|███████▉  | 315604/400000 [00:33<00:08, 9845.65it/s] 79%|███████▉  | 316590/400000 [00:34<00:08, 9602.33it/s] 79%|███████▉  | 317553/400000 [00:34<00:08, 9550.54it/s] 80%|███████▉  | 318510/400000 [00:34<00:09, 8964.73it/s] 80%|███████▉  | 319468/400000 [00:34<00:08, 9140.19it/s] 80%|████████  | 320389/400000 [00:34<00:08, 9058.17it/s] 80%|████████  | 321300/400000 [00:34<00:08, 9066.36it/s] 81%|████████  | 322279/400000 [00:34<00:08, 9270.37it/s] 81%|████████  | 323217/400000 [00:34<00:08, 9302.44it/s] 81%|████████  | 324150/400000 [00:34<00:08, 9126.70it/s] 81%|████████▏ | 325066/400000 [00:35<00:08, 8970.31it/s] 81%|████████▏ | 325966/400000 [00:35<00:08, 8928.31it/s] 82%|████████▏ | 326861/400000 [00:35<00:08, 8924.51it/s] 82%|████████▏ | 327755/400000 [00:35<00:08, 8761.96it/s] 82%|████████▏ | 328633/400000 [00:35<00:08, 8714.27it/s] 82%|████████▏ | 329506/400000 [00:35<00:08, 8394.64it/s] 83%|████████▎ | 330349/400000 [00:35<00:08, 8341.95it/s] 83%|████████▎ | 331218/400000 [00:35<00:08, 8443.38it/s] 83%|████████▎ | 332104/400000 [00:35<00:07, 8563.54it/s] 83%|████████▎ | 332963/400000 [00:35<00:08, 8310.83it/s] 83%|████████▎ | 333854/400000 [00:36<00:07, 8480.03it/s] 84%|████████▎ | 334705/400000 [00:36<00:07, 8446.68it/s] 84%|████████▍ | 335619/400000 [00:36<00:07, 8641.07it/s] 84%|████████▍ | 336510/400000 [00:36<00:07, 8718.83it/s] 84%|████████▍ | 337468/400000 [00:36<00:06, 8957.87it/s] 85%|████████▍ | 338467/400000 [00:36<00:06, 9242.87it/s] 85%|████████▍ | 339415/400000 [00:36<00:06, 9311.43it/s] 85%|████████▌ | 340370/400000 [00:36<00:06, 9381.49it/s] 85%|████████▌ | 341342/400000 [00:36<00:06, 9480.52it/s] 86%|████████▌ | 342292/400000 [00:36<00:06, 9235.64it/s] 86%|████████▌ | 343219/400000 [00:37<00:06, 9215.17it/s] 86%|████████▌ | 344143/400000 [00:37<00:06, 9050.99it/s] 86%|████████▋ | 345051/400000 [00:37<00:06, 8972.64it/s] 87%|████████▋ | 346036/400000 [00:37<00:05, 9217.44it/s] 87%|████████▋ | 347041/400000 [00:37<00:05, 9451.70it/s] 87%|████████▋ | 348005/400000 [00:37<00:05, 9506.53it/s] 87%|████████▋ | 348964/400000 [00:37<00:05, 9529.31it/s] 87%|████████▋ | 349919/400000 [00:37<00:05, 9144.03it/s] 88%|████████▊ | 350838/400000 [00:37<00:05, 8716.71it/s] 88%|████████▊ | 351885/400000 [00:38<00:05, 9177.06it/s] 88%|████████▊ | 352912/400000 [00:38<00:04, 9479.21it/s] 88%|████████▊ | 353871/400000 [00:38<00:04, 9383.50it/s] 89%|████████▊ | 354859/400000 [00:38<00:04, 9525.15it/s] 89%|████████▉ | 355889/400000 [00:38<00:04, 9743.00it/s] 89%|████████▉ | 356907/400000 [00:38<00:04, 9868.04it/s] 89%|████████▉ | 357904/400000 [00:38<00:04, 9895.40it/s] 90%|████████▉ | 358897/400000 [00:38<00:04, 9452.78it/s] 90%|████████▉ | 359849/400000 [00:38<00:04, 9314.92it/s] 90%|█████████ | 360786/400000 [00:38<00:04, 9266.60it/s] 90%|█████████ | 361717/400000 [00:39<00:04, 9271.24it/s] 91%|█████████ | 362749/400000 [00:39<00:03, 9559.92it/s] 91%|█████████ | 363735/400000 [00:39<00:03, 9647.61it/s] 91%|█████████ | 364787/400000 [00:39<00:03, 9892.58it/s] 91%|█████████▏| 365780/400000 [00:39<00:03, 9823.55it/s] 92%|█████████▏| 366765/400000 [00:39<00:03, 9650.93it/s] 92%|█████████▏| 367783/400000 [00:39<00:03, 9799.95it/s] 92%|█████████▏| 368766/400000 [00:39<00:03, 9722.12it/s] 92%|█████████▏| 369775/400000 [00:39<00:03, 9828.09it/s] 93%|█████████▎| 370799/400000 [00:39<00:02, 9946.80it/s] 93%|█████████▎| 371887/400000 [00:40<00:02, 10207.77it/s] 93%|█████████▎| 372937/400000 [00:40<00:02, 10291.84it/s] 93%|█████████▎| 373969/400000 [00:40<00:02, 10061.00it/s] 94%|█████████▍| 375048/400000 [00:40<00:02, 10267.59it/s] 94%|█████████▍| 376155/400000 [00:40<00:02, 10491.24it/s] 94%|█████████▍| 377249/400000 [00:40<00:02, 10619.94it/s] 95%|█████████▍| 378314/400000 [00:40<00:02, 10481.12it/s] 95%|█████████▍| 379365/400000 [00:40<00:02, 10293.33it/s] 95%|█████████▌| 380397/400000 [00:40<00:01, 10242.60it/s] 95%|█████████▌| 381425/400000 [00:40<00:01, 10252.89it/s] 96%|█████████▌| 382452/400000 [00:41<00:01, 10177.66it/s] 96%|█████████▌| 383476/400000 [00:41<00:01, 10193.75it/s] 96%|█████████▌| 384497/400000 [00:41<00:01, 9948.78it/s]  96%|█████████▋| 385507/400000 [00:41<00:01, 9991.67it/s] 97%|█████████▋| 386551/400000 [00:41<00:01, 10119.50it/s] 97%|█████████▋| 387565/400000 [00:41<00:01, 10044.98it/s] 97%|█████████▋| 388571/400000 [00:41<00:01, 9729.18it/s]  97%|█████████▋| 389547/400000 [00:41<00:01, 9691.94it/s] 98%|█████████▊| 390565/400000 [00:41<00:00, 9830.12it/s] 98%|█████████▊| 391588/400000 [00:42<00:00, 9945.93it/s] 98%|█████████▊| 392585/400000 [00:42<00:00, 9683.46it/s] 98%|█████████▊| 393557/400000 [00:42<00:00, 9577.71it/s] 99%|█████████▊| 394517/400000 [00:42<00:00, 9565.96it/s] 99%|█████████▉| 395570/400000 [00:42<00:00, 9833.33it/s] 99%|█████████▉| 396635/400000 [00:42<00:00, 10064.50it/s] 99%|█████████▉| 397645/400000 [00:42<00:00, 9745.74it/s] 100%|█████████▉| 398625/400000 [00:42<00:00, 9044.40it/s]100%|█████████▉| 399543/400000 [00:42<00:00, 8803.56it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9316.77it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f4e1d023160> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011060073140353092 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010918210382046907 	 Accuracy: 70

  model saves at 70% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15906 out of table with 15753 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


### Running {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} 'model_pars' 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 133, in benchmark_run
    return_ytrue=1)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 352, in predict
    ypred = model0(x_test)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 238, in forward
    emb_x = self.embed(x)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__
    result = self.forward(*input, **kwargs)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/nn/modules/sparse.py", line 114, in forward
    self.norm_type, self.scale_grad_by_freq, self.sparse)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/nn/functional.py", line 1467, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: index out of range: Tried to access index 15906 out of table with 15753 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Fit  ####################################################### 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-23 01:17:37.377202: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 01:17:37.381024: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-23 01:17:37.381144: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56547b6e7470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 01:17:37.381157: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4dc8689048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 9s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 6s - loss: 7.6896 - accuracy: 0.4985
 3000/25000 [==>...........................] - ETA: 5s - loss: 7.6155 - accuracy: 0.5033
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 4s - loss: 7.6329 - accuracy: 0.5022
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6027 - accuracy: 0.5042
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6168 - accuracy: 0.5033
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6329 - accuracy: 0.5022
11000/25000 [============>.................] - ETA: 3s - loss: 7.6429 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6491 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6369 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6378 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6786 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 6s 243us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


### Running {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'} {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'} 

  #### Setup Model   ############################################## 

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}} Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range 

  


### Running {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1} {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f4dc86a7908> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'} 

  #### Setup Model   ############################################## 
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 75)                0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 75, 40)            1720      
_________________________________________________________________
bidirectional_1 (Bidirection (None, 75, 100)           36400     
_________________________________________________________________
time_distributed_1 (TimeDist (None, 75, 50)            5050      
_________________________________________________________________
crf_1 (CRF)                  (None, 75, 5)             290       
=================================================================
Total params: 43,460
Trainable params: 43,460
Non-trainable params: 0
_________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f4d8c8d5cc0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 914ms/step - loss: 1.2983 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.2756 - val_crf_viterbi_accuracy: 0.6533

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text/ 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

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
NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_classifier.py", line 39, in <module>
    from util_transformer import (convert_examples_to_features, output_modes,
ModuleNotFoundError: No module named 'util_transformer'

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
NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
    output_path      = out_pars["model_path"]
KeyError: 'model_path'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textvae.py", line 51, in __init__
    texts, embeddings_index = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textvae.py", line 269, in get_dataset
    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
    file = builtins.open(filename, mode, buffering)
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 
